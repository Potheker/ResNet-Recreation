import torch
import torch.nn as nn
from network.network import ResNet
import data
import history
import sys


class Trainer:
    def __init__(self, net, device, criterion, optimizer, name="", scheduler=None):
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.name = name
        if scheduler:
            self.scheduler = scheduler
        else:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[91, 137], gamma=0.1)

    def train(self, epochs, train_loader, val_loader):

        # Data to be gathered
        train_loss = []
        train_er = []
        val_loss = []
        val_er = []

        print("\n---------------------------------------------------------------------")
        print(f"Beginning training on model {self.name} for {epochs} epochs.")
        print("---------------------------------------------------------------------\n")

        for epoch in range(epochs):

            # Train and gather data
            (tl, ter) = self.train_epoch(train_loader)
            train_loss.append(tl)
            train_er.append(ter)

            # Validate and gather data
            (vl, ver) = self.val_epoch(val_loader)
            val_loss.append(vl)
            val_er.append(ver)

            # Step lr scheduler
            self.scheduler.step()

            print(f"Progress on model {self.name}: Epoch {epoch+1:03d}/{epochs} train_loss: {train_loss[-1]:<19} train_error_rate: {train_er[-1]:<19} val_loss: {val_loss[-1]:<19} val_error_rate: {val_er[-1]:<19} lr: {self.scheduler.get_last_lr()}")

        return [train_loss, train_er, val_loss, val_er]

    def train_epoch(self, loader):

        # Data to be gathered
        total_loss = 0
        total_errors = 0

        for features, labels in loader:

            # Move to GPU
            features = features.to(self.device)
            labels = labels.to(self.device)

            # Run network
            result = self.net(features)

            # Calculate loss and optimize
            loss = self.criterion(result, labels)
            loss.backward()
            self.optimizer.step()

            # Reset grad
            self.optimizer.zero_grad()

            # Calculate errors and add up errors and loss
            _, prediction = torch.max(result, 1)
            total_errors += ((prediction == labels).sum().item())
            total_loss += loss.item()

        # Return loss and error rate
        return (total_loss*loader.batch_size/len(loader.dataset) , 1-total_errors/len(loader.dataset))

    def val_epoch(self, loader):
        with torch.no_grad():

            # Data to be gathered
            total_loss = 0
            total_errors = 0

            for features, labels in loader:

                # Move to GPU
                features = features.to(self.device)
                labels = labels.to(self.device)

                # Run network
                result = self.net(features)

                # Calculate loss and errors
                loss = self.criterion(result, labels)
                _, prediction = torch.max(result, 1)
                total_errors += ((prediction == labels).sum().item())
                total_loss += loss.item()

            # Return loss and error rate
            return (total_loss*loader.batch_size/len(loader.dataset) , 1-total_errors/len(loader.dataset))


# If run as a script, we parse network configurations and train accordingly
if __name__ == "__main__":    

    # List of all networks to be trained. Format of an entry: (n, res) with n being the amount of residual blocks and res being whether the network should be residual or not.
    network_parameters = []

    # Parse args
    for arg in sys.argv[1:]:
        sp = arg.split(",")
        if not len(sp) == 2 or not sp[0].isnumeric() or int(sp[0])<1 or not (sp[1] == "True" or sp[1] == "False"):
            raise Exception(f"Couldn't parse parameter \"{arg}\"! Please use the format \"n,res\", where n is a positive integer and res is True or False.")
        network_parameters.append((int(sp[0]), sp[1] == "True"))

    if not network_parameters:
        # If no specific parameters are passed, we iterate through every n from the paper and train all networks.
        print("No parameters were passed, so all networks will be trained. This will take significant time.")
        for i in (3,5,7,9):
            network_parameters.append((i, False))
            network_parameters.append((i, True))
    else:
        # If specific configurations are passed, only these are trained.
        print(f"{len(network_parameters)} configurations parsed. Training networks...")

    # Check if CUDA is available and set device accordingly
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    # Create data loaders
    (train_loader, val_loader, test_loader) = data.loaders()
    epochs = epochs = int(64000/len(train_loader))


    for i, (n, residual) in enumerate(network_parameters): 

        # Create network
        model = ResNet(n, residual).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)

        # Train model
        trainer = Trainer(net=model, device=device, criterion=criterion, optimizer=optimizer, name=f"{i+1}/{len(network_parameters)}")
        results = trainer.train(epochs=epochs, train_loader=train_loader, val_loader=val_loader)

        # Save model and history
        torch.save(model.state_dict(), f"./models/{n}_{residual}.pth")
        history.write_csv(n, residual, results)