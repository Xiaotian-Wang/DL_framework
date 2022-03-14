import torch

from models import Model
from trainer import Trainer
from record import Recorder
from inference import Predictor
import torchvision
import torchvision.transforms as transforms
import os
import yaml
from torch.utils.tensorboard import SummaryWriter


# Read the model configuration
with open('../config/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Initialize the model
model = Model()

# Define the optimizer
optimizer = torch.optim.AdamW(params=model.parameters())

# Select the device the model is trained on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the dataset and dataloader
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
# Define the loss function
criterion = torch.nn.CrossEntropyLoss()


# The Experiment class

class Experiment(object):

    def __init__(self, model=None, config=None, optimizer=None,
                 train_loader=None, val_loader=None, test_loader=None,
                 criterion=None, device=None, experiment_name='NewExperiment',
                 validation=False, record_training=False):
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.criterion = criterion
        self.device = device
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.experiment_name = experiment_name
        self.record_training =record_training

        self.saving_path = '../data/logs/' + self.experiment_name
        try:
            os.mkdir(self.saving_path)
        except FileExistsError:
            print('Directory Exists!')

        self.trainer = Trainer(model=self.model, optimizer=self.optimizer, train_loader=self.train_loader, criterion=self.criterion,
                  device=self.device, val_loader=self.val_loader, validation=validation, record=self.record_training, experiment_name=self.experiment_name)
        self.recorder = Recorder(model=self.model, experiment_name=self.experiment_name, config=self.config,
                 train_loader=self.train_loader)
        self.predictor = Predictor(model=self.model, data_loader=self.test_loader, device=self.device, criterion=self.criterion)


    def train(self):
        self.trainer.train()

    def test(self):
        return self.predictor.test()

    def save_result(self):
        self.recorder.save_result()


if __name__ == "__main__":
    exp = Experiment(model=model, optimizer=optimizer, train_loader=trainloader, val_loader=testloader, test_loader=testloader,
                 criterion=criterion, device=device, experiment_name="exp1", config=config, validation=True, record_training=True)
