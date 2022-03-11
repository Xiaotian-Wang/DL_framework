import os
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

class Recorder(object):
    def __init__(self, model=None, experiment_name=None, config=None,
                 train_loader=None):
        self.model = model
        self.experiment_name = experiment_name
        self.config = config
        self.train_loader = train_loader

    def save_result(self):
        saving_path = '../data/logs/' + self.experiment_name

        # TODO: Save the model, config and logs, testing result(if any)

        # Save the model
        os.mkdir(saving_path + '/models')
        torch.save(self.model, saving_path + '/models/model.pth')

        # Save the config info
        os.mkdir(saving_path + '/configs')
        with open(saving_path + '/configs/config.yaml', 'w+', encoding='utf-8') as f:
            yaml.dump(self.config, f)

        # Save the model structure
        writer = SummaryWriter(saving_path + '/models')
        dataiter = iter(self.train_loader)
        inputs, targets = dataiter.next()
        writer.add_graph(self.model, inputs)

