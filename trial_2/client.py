import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# Define the model architecture (same as in server.py)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  
        self.fc2 = nn.Linear(128, 64)   
        self.fc3 = nn.Linear(64, 10)      

    def forward(self, x):
        x = x.view(-1, 28*28)  
        x = F.relu(self.fc1(x))  
        x = F.relu(self.fc2(x))  
        x = self.fc3(x)  
        return F.log_softmax(x, dim=1)

# Define the Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, valloader):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.train_idx = 0
        self.epoch = 0

    def get_parameters(self):
        # return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        grad_collect = []
        for param in self.model.parameters():
            if param.requires_grad:
                # Copy the gradient if it exists
                if param.grad is not None:
                    # grad_collect.append(param.grad.clone())
                    grad = param.grad.clone()

                    grad_collect.append(grad)
                
        return grad_collect
    



    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        lr = config["lr"]
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        criterion = nn.NLLLoss()
        # epochs = 1
        # self.trainloader[self.train_idx]
        # for _ in range(epochs):
        #     for batch in tqdm(self.trainloader, "Training"):
        #         # print(batch.keys())
        #         images = batch[0]
        #         labels = batch[1]
        #     # for images, labels in self.trainloader:
        #         optimizer.zero_grad()
        #         outputs = self.model(images)
        #         loss = criterion(outputs, labels)
        #         loss.backward()
        #         optimizer.step()
        #         break

        
        # print(batch.keys())
        batch = None
        for i, b in enumerate(self.trainloader):
            if i == self.train_idx:
                batch = b
        # batch = self.trainlaoder[self.train_idx]
        # if batch == None:
        images = batch[0]
        labels = batch[1]
        optimizer.zero_grad()
        outputs = self.model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        # optimizer.step()
        self.train_idx +=1

        train_loss = loss.item()
        train_accuracy = (outputs.argmax(1) == labels).float().mean().item()

        if (self.train_idx == len(self.trainloader)):
            train_idx = 0
            self.epoch +=1

        # Return the model weights instead of gradients
        return self.get_parameters(), len(self.trainloader.dataset), {"train_loss": train_loss, "train_accuracy": train_accuracy, "done": self.epoch}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        criterion = nn.NLLLoss()
        loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.valloader:
                outputs = self.model(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        loss /= len(self.valloader.dataset)
        accuracy = correct / total
        
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}

def main():
    # Load the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trainset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    valset = datasets.MNIST("./data", train=False, transform=transform)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    valloader = DataLoader(valset, batch_size=32)

    # Create the model and the Flower client
    model = Net()
    client = FlowerClient(model, trainloader, valloader)

    # Start the Flower client
    fl.client.start_numpy_client(server_address = "localhost:8080", client = client.to_client())

if __name__ == "__main__":
    main()