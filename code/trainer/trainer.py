from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

class Trainer(object):
    def __init__(self, model=None, train_loader=None, val_loader=None, criterion=None, optimizer=None, device=None,
                 max_epochs=1000, validation=False, record=False, experiment_name='NewExperiment'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.max_epochs = max_epochs
        self.training_loss_list = []
        self.validation = validation
        self.record = record
        self.experiment_name = experiment_name
        if self.validation:
            self.val_loss_list = []
            self.val_outputs_list = []
            self.val_targets_list = []

        if self.record:
            self.saving_path = '../data/logs/'+self.experiment_name+'/training_logs'
            self.writer = SummaryWriter(log_dir=self.saving_path, flush_secs=1)
    def train(self):
        self.model = self.model.to(self.device)
        self.model.train()

        for epoch in tqdm(range(self.max_epochs)):
            running_loss = 0
            for i, data in enumerate(self.train_loader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, targets = data
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)

                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()*self.train_loader.batch_size

            if self.validation:
                val_running_loss = 0
                for j, val_data in enumerate(self.val_loader):
                    # get the inputs; data is a list of [inputs, targets]

                    val_inputs, val_targets = val_data
                    val_inputs = val_inputs.to(self.device)
                    val_targets = val_targets.to(self.device)

                    val_outputs = self.model(val_inputs)
                    self.val_outputs_list += [item.argmax() for item in val_outputs]
                    self.val_targets_list += val_targets

                    loss = self.criterion(val_outputs, val_targets)

                    val_running_loss += loss.item() * self.val_loader.batch_size
                val_loss = running_loss / self.val_loader.dataset.__len__()
                self.val_loss_list.append(val_loss)
                self.val_outputs_list = torch.tensor(self.val_outputs_list).tolist()
                self.val_targets_list = torch.tensor(self.val_targets_list).tolist()


            training_loss = running_loss/self.train_loader.dataset.__len__()
            self.training_loss_list.append(training_loss)
            if self.record:
                self.writer.add_scalars("Loss", {"train": training_loss}, epoch)
                if self.validation:
                    self.writer.add_scalars("Loss", {"validation": val_loss}, epoch)
            self.writer.flush()
            print(f"\n Epoch {epoch}/{self.max_epochs}, training loss {training_loss}")
