import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

# Define the model architecture (same as in client.py)
class Net(torch.nn.Module):
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
       
# Custom Flower Strategy
class FedCustom(fl.server.strategy.Strategy):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.global_model = Net()

    def __repr__(self) -> str:
        return "FedCustom"

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        # net = Net()
        # return fl.common.ndarrays_to_parameters(
        #     [val.cpu().numpy() for _, val in net.state_dict().items()]
        # )
        return fl.common.ndarrays_to_parameters(
            [val.cpu().numpy() for _, val in self.global_model.state_dict().items()]
        )

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Create custom configs
        n_clients = len(clients)
        half_clients = n_clients // 2
        standard_config = {"lr": 0.001}
        higher_lr_config = {"lr": 0.003}
        fit_configurations = []
        for idx, client in enumerate(clients):
            if idx < half_clients:
                fit_configurations.append((client, FitIns(parameters, standard_config)))
            else:
                fit_configurations.append(
                    (client, FitIns(parameters, standard_config))
                )
        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}

        # weights = [
        #     # parameters_to_ndarrays(result.parameters) for _, result in results
        #     parameters_to_ndarrays(result.parameters) for _, result in results
        # ]
        gradients = [
            parameters_to_ndarrays(result.parameters) for _, result in results
            # torch.tensor(result.parameters) for _, result in results
            # result.parameters for _, result in results
        ]

        done = results[0][1].metrics['done'] == 1

        # for _, result in results:


        # print(type(gradients[0][0]))
        # print(gradients[0][0])

        gradients_torch = [[torch.from_numpy(arr) for arr in sublist] for sublist in gradients]
        param_list = [torch.cat([xx.reshape(-1, 1) for xx in x], dim=0) for x in gradients_torch]
        # param_list = [np.concatenate([xx.reshape(-1, 1) for xx in x], axis=0) for x in gradients]

        mean_nd = torch.mean(torch.cat(param_list, dim=1), dim=-1)
        # concatenated_array = np.concatenate(param_list, axis=1)

        # mean_nd = np.mean(np.concatenate(param_list, axis=1), axis=-1)

        idx = 0
        global_model = self.get_global_model()
        for param in global_model.parameters():
            if param.requires_grad:
                # Determine the number of elements in the parameter tensor
                num_param_elements = param.numel()

                # Extract the corresponding slice from mean_nd
                param_update_slice = mean_nd[idx:idx + num_param_elements]

                # Reshape the slice to match the parameter's shape
                param_update = param_update_slice.view(param.size())

                # Update the parameter directly, using .data to avoid autograd tracking
                with torch.no_grad():  # Ensures the operation is not tracked by autograd
                    param.data -= 0.001 * param_update

                # Update idx to move to the next segment of mean_nd for the next parameter
                idx += num_param_elements





        # for _, result in results:
        #     print(f'RESULT GRADS: result.gradients') 


        # sizes = [result.num_examples for _, result in results]
        # total_size = sum(sizes)
        # print(f"Total number of examples: {total_size}")
        
        # # Compute the weighted average of model weights
        # avg_weights = [
        #     np.average([weights[i][j] for i in range(len(weights))], axis=0, weights=sizes)
        #     for j in range(len(weights[0]))
        # ]

        # Convert the averaged weights to a state dictionary
        # global_model = self.get_global_model()
        # state_dict = {k: torch.tensor(v) for k, v in zip(global_model.state_dict().keys(), avg_weights)}

        # Load the state dictionary into the global model
        # global_model.load_state_dict(state_dict)

        # Evaluate the updated global model
        # TODOODODODODOODODODDODD
        if done:
            self.evaluate_global_model(server_round)

        # Convert the updated global model parameters back to Parameters object
        updated_parameters = ndarrays_to_parameters(
            [val.cpu().numpy() for _, val in global_model.state_dict().items()]
        )

        # Return the updated global model parameters
        return updated_parameters, {}
    
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []
        config = {}
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""

        if not results:
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        metrics_aggregated = {}
        return loss_aggregated, metrics_aggregated
    

    def evaluate(
        self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate the global model parameters."""
        # net = Net()
        net = self.global_model

        ndarrays = parameters_to_ndarrays(parameters)
        state_dict = {k: torch.tensor(v) for k, v in zip(net.state_dict().keys(), ndarrays)}
        net.load_state_dict(state_dict)
        net.eval()

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        valset = datasets.MNIST("./data", train=False, download = True, transform=transform)
        valloader = DataLoader(valset, batch_size=32)

        criterion = torch.nn.NLLLoss()
        loss, correct = 0, 0
        with torch.no_grad():
            for images, labels in valloader:
                outputs = net(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        loss /= len(valloader.dataset)
        accuracy = correct / len(valloader.dataset)
        print()
        return loss, {"accuracy": accuracy}

    def evaluate_global_model(self, server_round):
        """Evaluate the global model."""
        parameters = self.get_global_model_parameters()
        loss, metrics = self.evaluate(server_round=server_round, parameters=parameters)
        print(f"Global Model - Average Loss: {loss:.4f}, Accuracy: {metrics['accuracy']:.4f}")

    def get_global_model_parameters(self):
        """Return the global model parameters."""
        # net = Net()
        net = self.global_model
        return fl.common.ndarrays_to_parameters(
            [val.cpu().numpy() for _, val in net.state_dict().items()]
        )
    
    def get_global_model(self):
        # return Net()
        return self.global_model

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

def main():
    # Define the Flower strategy
    strategy = FedCustom(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
    )
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trainset = datasets.MNIST("./data", train=True, download = True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=32)

    # Start the Flower server
    fl.server.start_server(
        server_address="localhost:8080",
        # config=fl.server.ServerConfig(num_rounds=10),
        config=fl.server.ServerConfig(num_rounds=len(trainloader)),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()