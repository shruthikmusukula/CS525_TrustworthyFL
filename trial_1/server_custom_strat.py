from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics

import CustomFedAvg as CFA

import model

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# # Define strategy
# strategy = CustomFedAvg(evaluate_metrics_aggregation_fn=weighted_average)
global_model = model.Model()

def get_weights(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=CFA.CustomFedAvg(initial_parameters=fl.common.ndarrays_to_parameters(get_weights(global_model))),
)