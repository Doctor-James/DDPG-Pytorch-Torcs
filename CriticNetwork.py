import torch
import torch.nn as nn
class CriticNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__(self)
        self.mlp_s1 = nn.Sequential(nn.Linear(in_features=state_size,out_features=300),
                                 nn.ReLU(inplace=True),

                                 )
        self.mlp_a1 = nn.Sequential(nn.Linear(in_features=action_size, out_features=600),
                                 # nn.ReLU(inplace=True),
                                 )
        self.mlp_s2 = nn.Sequential(nn.Linear(in_features=300, out_features=600),
                                 # nn.ReLU(inplace=True),
                                 )
        self.mlp_as1 = nn.Sequential(nn.Linear(in_features=600, out_features=600),
                                 nn.ReLU(inplace=True),
                                 )
        self.mlp_as2 = nn.Sequential(nn.Linear(in_features=600, out_features=action_size),
                                 # nn.ReLU(inplace=True),
                                 )

    def forward(self, state, action):
        state = torch.tensor(state)
        action = torch.tensor(action)
        s1 = self.mlp_s1(state)
        s2 = self.mlp_s2(s1)
        a1 = self.mlp_a1(action)
        sa = torch.sum(s2,a1)
        sa = self.mlp_as1(sa)
        Value = self.mlp_as2(sa)
        return Value




