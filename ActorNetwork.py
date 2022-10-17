import torch
import torch.nn as nn
class ActorNetwork(nn.Module):
    def __init__(self, state_size):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_features=state_size,out_features=300),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(in_features=300, out_features=600),
                                 nn.ReLU(inplace=True),
                                 ).to(torch.float64)
        self.Steering = nn.Sequential(nn.Linear(in_features=600,out_features=1),
                                 nn.Tanh()
                                 ).to(torch.float64)
        self.Acceleration = nn.Sequential(nn.Linear(in_features=600,out_features=1),
                                 nn.Sigmoid()
                                 ).to(torch.float64)
        self.Brake = nn.Sequential(nn.Linear(in_features=600,out_features=1),
                                 nn.Sigmoid()
                                 ).to(torch.float64)

    def forward(self, x):
        x = torch.tensor(x)
        x = self.mlp(x)
        Steering = self.Steering(x)
        Acceleration = self.Acceleration(x)
        Brake = self.Brake(x)
        y = torch.cat((Steering,Acceleration,Brake),dim=1)
        return y

    # target_network
    def target_train(self,actor,TAU):
        for name in self.state_dict():
            self.state_dict()[name][0] = TAU * actor.state_dict()[name][0] + (1 - TAU) * self.state_dict()[name][0]




