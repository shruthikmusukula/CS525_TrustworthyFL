from typing import List, Tuple

import flwr as fl
from flwr_datasets import FederatedDataset
from flwr.common import Metrics

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, ToTensor
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
import torch
import model as model
from tqdm import tqdm

DEVICE = "cpu"
DATASET_NAME = "mnist"
BATCH_SIZE = 32
NUM_EPOCHS=1
NUM_CLIENTS = 3

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def get_parameters(model):
         

    gradients = []
    for _, param in model.named_parameters():
        if param.grad is not None:
            gradients.append(param.grad.cpu().numpy())

    return gradients
# FedAvg Strategy
# strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)


# Krum Strategy
# strategy = fl.server.strategy.Krum(evaluate_metrics_aggregation_fn=weighted_average)

# Pass parameters to the Strategy for server-side parameter initialization

def test(model, val_loader, criterion):
    '''
    Validation step in our pipeline, for model to reinforce against new dataset (stray away from overfitting)
    '''
    model.eval()
    # criterion  
    val_loss = 0
    correct = 0
    with torch.no_grad():  
        for batch in tqdm(val_loader, "Testing"):
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


trainloader, valloader, testloader = load_dataset(cid=0)

def set_parameters(model, gradients):
    # params_dict = zip(model.state_dict().keys(), parameters)
    # state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    # model.load_state_dict(state_dict, strict=True)
    for (name, param), grad in zip(model.named_parameters(), gradients):
        if grad is not None:
            param.grad = torch.from_numpy(grad).to(param.device)
        else:
            # Handle the case where some gradients might be None or missing.
            # This might involve setting gradients to zeros, or you might choose
            # to leave them as None, depending on your update strategy.
            # Example for setting to zeros:
            param.grad = torch.zeros_like(param)


def get_weights(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

global_model = model.Model()
criterion = nn.NLLLoss() 
# params = get_weights(model.Model())

def evaluate(
    server_round: int,
    parameters: fl.common.NDArrays,
    config: Dict[str, fl.common.Scalar],
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    net = global_model
    set_parameters(net, parameters)  # Update model with the latest parameters
    loss, accuracy = test(net, testloader, criterion=criterion)
    print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
    return loss, {"accuracy": accuracy}

strategy = fl.server.strategy.FedAvg(
    initial_parameters=fl.common.ndarrays_to_parameters(get_weights(global_model)),
    evaluate_fn=evaluate,
)



# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
)