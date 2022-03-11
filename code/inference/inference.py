import torch

class Predictor(object):
    def __init__(self, model=None, data_loader=None, device=None, criterion=None):
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.criterion = criterion
        model.eval()
        self.outputs_list = []
        self.targets_list = []
        self.testing_loss = None

    def test(self):
        running_loss = 0
        for i, data in enumerate(self.data_loader):
            # get the inputs; data is a list of [inputs, targets]

            inputs, targets = data
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(inputs)
            self.outputs_list += [item.argmax() for item in outputs]
            self.targets_list += targets

            loss = self.criterion(outputs, targets)

            running_loss += loss.item() * self.data_loader.batch_size
        self.testing_loss = running_loss / self.data_loader.dataset.__len__()
        self.outputs_list = torch.tensor(self.outputs_list).tolist()
        self.targets_list = torch.tensor(self.targets_list).tolist()
        return self.testing_loss

