import torch
import torch.nn as nn
class ActorNetwork(nn.Module):
    def __init__(self, state_size):
        super().__init__(self)
        self.mlp = nn.Sequential(nn.Linear(in_features=state_size,out_features=300),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(in_features=300, out_features=600),
                                 nn.ReLU(inplace=True),
                                 )
        self.Steering = nn.Sequential(nn.Linear(in_features=600,out_features=1),
                                 nn.Tanh(inplace=True)
                                 )
        self.Acceleration = nn.Sequential(nn.Linear(in_features=600,out_features=1),
                                 nn.Sigmoid(inplace=True)
                                 )
        self.Brake = nn.Sequential(nn.Linear(in_features=600,out_features=1),
                                 nn.Sigmoid(inplace=True)
                                 )

    def forward(self, x):
        x = torch.tensor(x)
        x = self.mlp(x)
        Steering = self.Steering(x)
        Acceleration = self.Acceleration(x)
        Brake = self.Brake(x)
        y = torch.cat((Steering,Acceleration,Brake),dim=0)
        return y




