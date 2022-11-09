import numpy as np
import torch

import state_engine

class MCTSNode():
    """
    Search tree node.

    Parameters Attributes
    ---------------------
    n_actions: int

    Location Attributes
    -------------------
    index: int
    parent_index: int
    action_index: int
    branch_indices: list(int)
    descendants: set(int)

    Data Attributes
    ---------------
    state: state_engine.Position
    value: float
    action_probabilities: numpy.ndarray(float)
    visits: list(int)
    branch_path_values: numpy.array(float)
    branch_visits: numpy.ndarray(int)
    ucb: numpy.ndarray(float)
    """
    def __init__(self, index, parent_index, action_index, state):
        # location
        self.index = index
        self.parent_index = parent_index
        self.action_index = action_index
        self.branch_indices = []
        self.descendants = set()
        # data
        self.state = state
        self.value = 0.0
        self.action_probabilities = np.array([])
        self.visits = 0
        self.path_value = 0.0
        self.branch_path_values = []
        self.branch_visits = np.array([])
        self.ucb = []

    def calculate_ucb(self, c_puct=1.0):
        """
        Vector computation of the Upper Confidence Bound 'UCB = Q + U', used to rank choices.
        """
        # Q: mean action value; zero-divisions set to zero
        mav = np.divide(self.branch_path_values, self.branch_visits, out=np.zeros_like(self.branch_path_values), where=self.branch_visits!=0)
        # U: exploration value
        exv = np.divide(c_puct * self.action_probabilities + np.sqrt(self.visits), (1 + self.branch_visits))
        self.ucb = np.add(mav,exv) * self.state.to_play # sign from current player's perspective

class MCSTree():
    """
    A Monte Carlo Search Tree, integrating an inference model.
    """
    def __init__(self, model=None, n_sims=100, n_actions=362, batch_size=48):
        # parameters
        self.n_actions = n_actions
        self.n_sims = n_sims
        self.batch_size = batch_size
        self.noise_α = 3e-2
        self.noise_ε = 0.25
        self.temperature = 0.1 # used to transform visit counts into a probabilistic distribution

        # data
        self.model = model
        self.tree = dict()
        self.search_probabilities = np.zeros(shape=n_actions, dtype=np.float32)

        # initialize root node
        self.add_node(index=0, parent_index=None, action_index=None, state=state_engine.Position())
        self.set_root(index=0)
        self.next_index = 1

        # expand root node
        input_states = torch.zeros((1,*state_engine.DATA_SHAPE))
        input_states[0] = torch.from_numpy(state_engine.get_history(self.root.state))
        probabilities,values = self.eval_states(input_states)
        self.expand_node(self.root, probabilities[0], values[0])

    def reset(self):
        """
        Resets the search tree to a newly-initialized state for a new game.
        """
        self.__init__(self.model, n_sims=self.n_sims, n_actions=self.n_actions, batch_size=self.batch_size)

    def __len__(self, idx:int):
        """
        Returns the number of initalized nodes in the search tree.
        """
        return len(self.tree)

    def __getitem__(self, idx:int):
        """
        Returns the `MCTSNode` associated with the index `idx`.
        """
        if idx in self.tree:
            return self.tree[idx]
        else:
            return None

    def backpropagate_statistics(self, index):
        """
        Updates search statistics (value and visit count) back along a nodes path to root.

        Notes
        -----
        Values alternate sign according to perspective. A node will see the values of its branch 
        nodes as sign-flipped. This accounts for alternating player turns.

        The branch path values and path value of a node are signed from its perspective.
        """
        node = self.tree[index]
        value = node.value
        while node.index != self.root.index:
            # update current node
            node.path_value += value
            node.visits += 1
            # update parent node's record of current node
            parent_node = self.tree[node.parent_index]
            parent_node.branch_path_values[node.action_index] += value * -1
            parent_node.branch_visits[node.action_index] += 1

            value *= -1 # flip sign to opposite player perspective
            node = parent_node
            node.descendants.add(index) # avoids setting a node as its own descendant; includes root
        node.path_value += value
        node.visits += 1

    def add_node(self, index, parent_index, action_index, state):
        """
        Initializes and adds an empty node to the search tree. Nodes must be expanded/'explored' to be 'useful'.
        """
        node = MCTSNode(index = index, parent_index=parent_index, action_index=action_index, state=state)
        self.tree[index] = node
    
    def expand_node(self, node, probabilities, value, state=None):
        """
        Expands/explores a node by populating its statistics and adding its edges to the search tree.

        Parameters
        ----------
        node: MCTSNode
        probailities: numpy.ndarray(float) or torch.Tensor(float)
        value: float
        state: (optional) state_engine.Position
            Optionality allows `expand_node` to be used in 3 cases: 1) root node initialization, 
            2) simulation expansion, 3) explicit move selection.
        """
        # node data statistics
        node.value = value
        node.action_probabilities = probabilities
        node.state = state if state is not None else node.state
        # update node edges into tree
        node.branch_indices = np.array([self.next_index + i for i in range(self.n_actions)])
        self.next_index += self.n_actions
        node.descendants.update(node.branch_indices)
        for i,branch_index in enumerate(node.branch_indices):
            self.add_node(branch_index, parent_index=node.index, action_index=i, state=None)
        # node edge data statistics
        node.branch_path_values = np.array([0.0]*self.n_actions)
        node.branch_visits = np.array([0]*self.n_actions)

    def eval_states(self, input_states):
        """
        Runs inference on a batch of state histories, returning associated action probabilities and 
        values. Decoupling this method from `expand_node` allows simulations to be run in batches.

        Input shape: (batch_size, data_shape). Input shape for inference of a single Go state: (1, 17, 19, 19)
        """
        # send data to GPU if model on GPU; turn off autograd for inference
        with torch.no_grad():
            device = next(self.model.parameters()).device # type: ignore 
            if device.type == 'cuda':
                input_states = input_states.cuda()
            probabilities, value = self.model(input_states) # type: ignore 
        # apply nose: P(s) = (1 - ε) * probabilities + ε * η; η = dirichlet(α)
        dirichlet_noise = torch.distributions.Dirichlet(self.noise_α * torch.ones_like(probabilities)).sample()
        probabilities = (1 - self.noise_ε) * probabilities + self.noise_ε * dirichlet_noise

        # nove output tensors back to CPU
        if probabilities.is_cuda or value.is_cuda:
            probabilities = probabilities.cpu()
            value = value.cpu()

        return probabilities, value

    def set_root(self, index):
        """
        Sets the root of the search tree to the provided index.
        """
        self.root = self.tree[index]

    def prune_substree(self, index):
        """
        Removes a node and its subtree from the search tree.
        """
        for descendant in self.tree[index].descendants:
            del self.tree[descendant]
        del self.tree[index]

    def simulate(self, runs=1):
        """
        Runs a simulation and updates node statistics.

        Notes
        -----
        To increase performance in Python without parallel access to shared data, this method 
        queues state inferences in batches for the GPU.
        """
        eval_queue = set() # mutex
        check_index_in_queue = np.vectorize(lambda index: index in eval_queue)

        # explore tree and add new nodes to batch eval queue
        for i in range(runs):
            # explore tree until an end or unexplored reached
            node = self.root
            while node.state is not None and not node.state.is_game_over():
                # mask invalid and queued edges
                mask = np.zeros(self.n_actions, dtype=np.uint8)
                mask[np.where(node.state.all_legal_moves()==0)] = 1
                mask |= check_index_in_queue(node.branch_indices)
                if mask.mean() == 1.0: # no valid or unqueued moves available
                    break
                # select next node by UCB
                node.calculate_ucb()
                action_index = np.argmax(np.ma.array(node.ucb, mask=mask))
                node = self.tree[node.branch_indices[action_index]]

            # add unexplored node to evaluation queue
            if node.state is None:
                eval_queue.add(node.index)
            # backpropagate statistics for existing leaf nodes (game overs)
            else:
                self.backpropagate_statistics(node.index)

        # evaluate nodes in queue
        eval_queue = list(eval_queue)
        if len(eval_queue) > 0:
            input_states = torch.zeros((len(eval_queue), *state_engine.DATA_SHAPE))
            for i,index in enumerate(eval_queue): # build input data batch
                # get node state
                node = self.tree[index]
                parent_node = self.tree[self.tree[index].parent_index]
                node.state = parent_node.state.play_move(c=state_engine.action_array_index_to_coord(node.action_index))
                input_states[i] = torch.from_numpy(state_engine.get_history(node.state))
            probabilities,values = self.eval_states(input_states) # batch inference
            for i,index in enumerate(eval_queue): # update nodes
                self.expand_node(self.tree[index], probabilities[i], values[i]) # fully-initializes a node
            
            # backpropagate statistics for newly evaluated leaf nodes:
            for index in eval_queue:
                self.backpropagate_statistics(index)

    def search(self):
        """
        Runs Monte Carlo Tree Search and updates the tree's search probabilities ("policy").
        """
        self.batch_size = 1 if self.batch_size < 1 else self.batch_size

        # run simulations
        sims_remaining = self.n_sims
        while sims_remaining:
            runs = min(sims_remaining, self.batch_size)
            self.simulate(runs=runs)
            sims_remaining = max(0, sims_remaining - self.batch_size)
        
        # build search probabilities
        self.search_probabilities = self.root.branch_visits ** (1./self.temperature)
        if self.search_probabilities.sum() == 0: # if there're problems w/ zero-divisions on small temperatures, set argmax(visits) to 1
            argmax_visits = np.argmax(self.root.branch_visits)
            self.search_probabilities[argmax_visits] = 1.0
        self.search_probabilities /= self.search_probabilities.sum()

    def play_move(self, move_index=None):
        """
        Plays one move and updates the tree. Will run MCTS to select a move if none provided.
        """
        if move_index is None:
            self.search()
            if self.root.state.n < 30: # probabilistically-stochastic argmax in early moves
                move_index = np.random.choice(list(range(len(self.search_probabilities))), p=self.search_probabilities)
            else: # deterministically select most visited edge
                move_index = np.argmax(self.search_probabilities)

        new_root_index = self.root.branch_indices[move_index]

        # expand node if unexplored (wasn't part of MCTS)
        if self.tree[new_root_index].state is None:
            coordinate = state_engine.action_array_index_to_coord(move_index) # type: ignore 
            state = self.root.state.play_move(c=coordinate)
            input_state = torch.zeros((1,*state_engine.DATA_SHAPE))
            input_state[0] = torch.from_numpy(state_engine.get_history(state))
            probabilities,value = self.eval_states(input_state)
            self.expand_node(self.tree[new_root_index], probabilities[0], value[0], state)
        
        # play move, update tree
        for index in self.root.branch_indices:
            if index != new_root_index:
                self.prune_substree(index)
        del self.tree[self.root.index]
        self.root = self.tree[new_root_index]

        # returning played move simplifies self-play code
        return move_index
