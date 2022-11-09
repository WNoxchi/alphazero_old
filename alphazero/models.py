import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, planes=256):
        """Architecture Definition"""
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(num_features=planes)
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(num_features=planes)

    def forward(self, x:torch.Tensor):
        """Computation Definition"""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # skip connection
        out += identity
        out = F.relu(out)

        return out

class PolicyHead(nn.Module):
    def __init__(self, n_actions=362):
        """Architecture Definition"""
        super().__init__()
        self.conv   = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1, stride=1)
        self.bn     = nn.BatchNorm2d(num_features=2)
        self.linear = nn.Linear(in_features=19*19*2, out_features=n_actions)

    def forward(self, x:torch.Tensor):
        """Computation Definition"""
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)

        # nn.AdaptivePool2d((1,1)) needed for variable-sized input
        # x = x.view(-1, 2*19*19) # batch_size x channgels x board_rows x board_cols
        x = torch.flatten(x, 1)

        x = self.linear(x)
        # x = F.softmax(x, dim=1) #or logsoftmax(x).exp() ? | do not use softmax if using Cross Entropy loss (log softmax already included)
        return x

class ValueHead(nn.Module):
    def __init__(self):
        """Architecture Definition"""
        super().__init__()
        self.conv = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(num_features=1)
        self.linear1 = nn.Linear(in_features=19*19*1, out_features=256)
        self.linear2 = nn.Linear(in_features=256, out_features=256)
        self.linear3 = nn.Linear(in_features=256, out_features=1)

    def forward(self, x):
        """Computation Definition"""
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)

        # nn.AdaptivePool2d((1,1)) needed for variable-sized input
        # x = x.view(-1, 2*19*19) # batch_size x channgels x board_rows x board_cols
        x = torch.flatten(x, 1)

        x = self.linear1(x)
        x = self.linear2(x)
        x = F.relu(x)

        x = self.linear3(x)
        x = torch.tanh(x)
        return x

class GoNet(nn.Module):
    def __init__(self, res_blocks=19, n_actions=362):
        """Architecture Definition"""
        super().__init__()
        self.conv = nn.Conv2d(in_channels=17, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn   = nn.BatchNorm2d(num_features=256)
        self.res_stem   = nn.Sequential(*[ResBlock(planes=256) for i in range(res_blocks)])
        self.policy_out = PolicyHead(n_actions=n_actions)
        self.value_out  = ValueHead()

    def forward(self, x):
        """Computation Definition"""
        # 1 conv block
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)

        # res blocks
        x = self.res_stem(x)

        # policy head
        p = self.policy_out(x)
        # value head
        v = self.value_out(x)

        return p,v
