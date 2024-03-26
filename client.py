import argparse
import warnings
from collections import OrderedDict

import flwr as fl
from flwr_datasets import FederatedDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

import torch.optim as optim
import torchvision.transforms as transforms


warnings.filterwarnings("ignore", category=UserWarning)
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"
# NUM_CLIENTS = 3
DATASET_NAME = "mnist"
BATCH_SIZE = 32
NUM_EPOCHS=1

# class Net(nn.Module):
#     """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

#     def __init__(self) -> None:
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)

class ClientModel(nn.Module):
    '''
    Created a Client Model which goes through 3 layers
    - First layer is a linear layer based on the dimension of the mnist images (28x28)
    - Second layer takes the output of the first in order to break down dimensions
    - Final layer looks to output a number bw 0-9 i.e. 10 outputs
    
    '''
    def __init__(self):
        super(ClientModel, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  
        self.fc2 = nn.Linear(128, 64)   
        self.fc3 = nn.Linear(64, 10)      

    def forward(self, x):
        x = x.view(-1, 28*28)  
        x = F.relu(self.fc1(x))  
        x = F.relu(self.fc2(x))  
        x = self.fc3(x)  
        return F.log_softmax(x, dim=1)  # Return log-probability
    



# def train(net, trainloader, epochs):
#     """Train the model on the training set."""
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#     for _ in range(epochs):
#         for batch in tqdm(trainloader, "Training"):
#             images = batch["img"]
#             labels = batch["label"]
#             optimizer.zero_grad()
#             criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
#             optimizer.step()

def train(model, train_loader, epochs, criterion, optimizer):
    '''
     Basic train function which takes a single partitioned dataset, 
     and is able to look at batched output, loss, and step in the right direction.
     Might want to implement tqdm library
    '''

    # criterion = nn.NLLLoss()  
    # optimizer = optim.Adam(model.parameters(), lr=0.001) 
    model.train()  
    batch_idx = 0
    for epoch in range(epochs):
        for batch in tqdm(trainloader, "Training"):
            data = batch["image"]
            target = batch["label"]

            optimizer.zero_grad() 
            output = model(data) 
            loss = criterion(output, target)  
            loss.backward()  
            optimizer.step()  
        
            if batch_idx % 100 == 0:  # Log training status every 100 batches
                print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
            batch_idx +=1
 


# def test(net, testloader):
#     """Validate the model on the test set."""
#     criterion = torch.nn.CrossEntropyLoss()
#     correct, loss = 0, 0.0
#     with torch.no_grad():
#         for batch in tqdm(testloader, "Testing"):
#             images = batch["img"].to(DEVICE)
#             labels = batch["label"].to(DEVICE)
#             outputs = net(images)
#             loss += criterion(outputs, labels).item()
#             correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
#     accuracy = correct / len(testloader.dataset)
#     return loss, accuracy

def validate(model, val_loader, criterion):
    '''
    Validation step in our pipeline, for model to reinforce against new dataset (stray away from overfitting)
    '''
    model.eval()
    # criterion  
    val_loss = 0
    correct = 0
    with torch.no_grad():  
        for batch in tqdm(val_loader, "Validation"):
            data = batch["image"]
            target = batch["label"]

            output = model(data)
            val_loss += criterion(output, target).item()  
            pred = output.argmax(dim=1, keepdim=True)  
            correct += pred.eq(target.view_as(pred)).sum().item()

    # val_loss /= len(val_loader.dataset)
    print(f'\nValidation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(val_loader.dataset)} ({100. * correct / len(val_loader.dataset):.0f}%)\n')
    accuracy = correct / len(val_loader.dataset)
    return val_loss, accuracy

# def load_data(partition_id):
#     """Load partition CIFAR10 data."""
#     fds = FederatedDataset(dataset="cifar10", partitioners={"train": 3})
#     partition = fds.load_partition(partition_id)
#     # Divide data on each node: 80% train, 20% test
#     partition_train_test = partition.train_test_split(test_size=0.2)
#     pytorch_transforms = Compose(
#         [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
#     )

#     def apply_transforms(batch):
#         """Apply transforms to the partition from FederatedDataset."""
#         batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
#         return batch

#     partition_train_test = partition_train_test.with_transform(apply_transforms)
#     trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
#     testloader = DataLoader(partition_train_test["test"], batch_size=32)
#     return trainloader, testloader

def load_dataset(cid):
        '''Loading DATASET_NAME (mnist) for given cid'''
        fds = FederatedDataset(dataset=DATASET_NAME, partitioners={"train": NUM_CLIENTS})
        print(fds)


        def apply_transforms(batch):

            transform = transforms.Compose(
                [
                    transforms.ToTensor(), 
                    transforms.Normalize((0.5,), (0.5,))  
                ]
            )
            batch["image"] = [transform(img) for img in batch["image"]]
            return batch

       
        partition = fds.load_partition(cid, "train")
        partition = partition.with_transform(apply_transforms)
        partition = partition.train_test_split(train_size=0.8)
        trainloader = DataLoader(partition["train"], batch_size=BATCH_SIZE, drop_last=True)
        valloader = DataLoader(partition["test"], batch_size=BATCH_SIZE, drop_last=True)
        
        testset = fds.load_full("test").with_transform(apply_transforms)
        testloader = DataLoader(testset, batch_size=BATCH_SIZE,drop_last=True)
        return trainloader, valloader, testloader

# Get partition id
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "-p",
    "--partition-id",
    choices=[0, 1, 2],
    required=True,
    type=int,
    help="Partition of the dataset divided into 3 iid partitions created artificially.",
)

# Get number of clients
parser.add_argument(
    "-n",
    "--nclients",
    choices=range(1,21),
    metavar="[1-20]",
    required=True,
    type=int,
    help="Number of client processes for the current simulation.",
)

# Get main server IP address
parser.add_argument(
    "-s",
    "--server-ip",
    required=True,
    type=str,
    help="IP Address of the main FL server for the current simulation.",
)

# Parse all CLI arguments
cid = parser.parse_args().partition_id
NUM_CLIENTS = parser.parse_args().nclients
server_address = parser.parse_args().server_ip

# Load model and data (simple CNN, CIFAR-10)
model= ClientModel()
# trainloader, testloader = load_data(partition_id=partition_id)
trainloader, valloader, _ = load_dataset(cid=cid)
criterion = nn.NLLLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
         
        # if cid % 2 == 0:
        #     # Generate random parameters
        #     random_parameters = [torch.rand_like(param).numpy() for _, param in model.state_dict().items()]
        #     return random_parameters
        # else:
        #     # Return the actual model parameters
        #     return [param.cpu().numpy() for _, param in model.state_dict().items()]

        return [val.cpu().numpy() for _, val in model.state_dict().items()]
        # accessing grads
        # gradients = []
        # for _, param in model.named_parameters():
        #     if param.grad is not None:
        #         gradients.append(param.grad.cpu().numpy())

        # return gradients

    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(model, trainloader, epochs=NUM_EPOCHS, criterion=criterion, optimizer=optimizer)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = validate(model, valloader, criterion=criterion)
        return loss, len(valloader.dataset), {"accuracy": accuracy}


# Start Flower client
fl.client.start_client(
    server_address=server_address,
    client=FlowerClient().to_client(),
)