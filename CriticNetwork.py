import torch
import torch.nn as nn
class CriticNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.mlp_s1 = nn.Sequential(nn.Linear(in_features=state_size,out_features=300),
                                 nn.ReLU(inplace=True),
                                 ).to(torch.float64)
        self.mlp_a1 = nn.Sequential(nn.Linear(in_features=action_size, out_features=600),
                                 # nn.ReLU(inplace=True),
                                 ).to(torch.float64)
        self.mlp_s2 = nn.Sequential(nn.Linear(in_features=300, out_features=600),
                                 # nn.ReLU(inplace=True),
                                 ).to(torch.float64)
        self.mlp_as1 = nn.Sequential(nn.Linear(in_features=600, out_features=600),
                                 nn.ReLU(inplace=True),
                                 ).to(torch.float64)
        self.mlp_as2 = nn.Sequential(nn.Linear(in_features=600, out_features=action_size),
                                 # nn.ReLU(inplace=True),
                                 ).to(torch.float64)
        self.device = "cuda:0"
    def forward(self, state, action):
        state = torch.tensor(state).to(self.device)
        action = torch.tensor(action).to(self.device)
        s1 = self.mlp_s1(state)
        s2 = self.mlp_s2(s1)
        a1 = self.mlp_a1(action)
        sa = self.mlp_as1((s2+a1))
        Value = self.mlp_as2(sa)
        return Value

    # target_network
    def target_train(self,critic,TAU):
        for name in self.state_dict():
            self.state_dict()[name][0] = TAU * critic.state_dict()[name][0] + (1 - TAU) * self.state_dict()[name][0]






