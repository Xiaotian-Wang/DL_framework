from torch.nn import Module
import torchvision
import torch

class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer = torchvision.models.resnet18()
        self.linear1 = torch.nn.Linear(1000, 512)
        self.linear2 = torch.nn.Linear(512, 128)
        self.linear3 = torch.nn.Linear(128, 10)
        self.activation = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten(1)
    def forward(self, image):

        x = self.layer(image/256)
        x = self.flatten(x)
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.linear3(x)
        return x
