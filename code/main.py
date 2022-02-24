import torch

from models import Model
from trainer import Trainer
import torchvision
import torchvision.transforms as transforms
import os

model = Model()
optimizer = torch.optim.AdamW(params=model.parameters())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='../data/datasets', train=True,
                                            download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                              shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../data/datasets', train=False,
                                           download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=2)

criterion = torch.nn.CrossEntropyLoss()

"""
trainer = Trainer(model=model, optimizer=optimizer, train_loader=trainloader, criterion=criterion,
                  device=device)
"""


class Experiment(object):

    def __init__(self, model=None, config=None, optimizer=None, train_loader=None, val_loader=None,
                 criterion=None, device=None, experiment_name=None):
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.criterion = criterion
        self.device = device
        self.val_loader = val_loader
        self.trainer = Trainer(model=self.model, optimizer=self.optimizer, train_loader=self.train_loader, criterion=self.criterion,
                  device=self.device)
        self.experiment_name = experiment_name

    def train(self, validation=False):
        self.trainer.train(validation=validation)

    def test(self, test_loader):
        # TODO: test the model on testing set, save the result to self
        pass

    def save_result(self):
        saving_path = '../data/logs/'+self.experiment_name
        try:
            os.mkdir(saving_path)
        except FileExistsError:
            print('Directory Exists!')
        else:
            # TODO: Save the model, config and logs, testing result(if any)
            pass
