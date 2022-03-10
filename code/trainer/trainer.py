from tqdm import tqdm


class Trainer(object):
    def __init__(self, model=None, train_loader=None, val_loader=None, criterion=None, optimizer=None, device=None,
                 max_epochs=1000):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.max_epochs = max_epochs
        self.loss_list = []

    def train(self, validation=False):
        self.model = self.model.to(self.device)
        self.model.train()

        for epoch in tqdm(range(self.max_epochs)):
            running_loss = 0
            for i, data in enumerate(self.train_loader):
                # get the inputs; data is a list of [inputs, labels]
                data = data
                inputs, targets = data
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)

                self.loss = self.criterion(outputs, targets)
                self.loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += self.loss.item()*self.train_loader.batch_size
                training_loss = running_loss/self.train_loader.dataset.__len__()
            self.loss_list.append(training_loss)
            print(f"\n Epoch {epoch}/{self.max_epochs}, training loss {training_loss}")
