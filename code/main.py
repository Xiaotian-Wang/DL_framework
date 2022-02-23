import torch

from models import Model
from trainer import Trainer
import torchvision
import torchvision.transforms as transforms

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

trainer = Trainer(model=model, optimizer=optimizer, train_loader=trainloader, criterion=criterion,
                  device=device)