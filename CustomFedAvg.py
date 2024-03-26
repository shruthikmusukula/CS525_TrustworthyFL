from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from flwr.common import (
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

import flwr as fl

from flwr.server.client_proxy import ClientProxy

class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, initial_parameters, *args, **kwargs):
        self.current_weights = parameters_to_ndarrays(initial_parameters)
        # now intitialize the parent FedAvg
        super().__init__(*args, **kwargs)

    def custom_aggregate(self, results: List[Tuple[NDArrays, int]]) -> NDArrays:
        num_examples_total = [result.num_examples for _, result in results]
        num_examples_total = sum(num_examples_total)

        # num_examples_total = (parameters_to_ndarrays(result.parameters) for _, result in results)
        # num_examples_total = sum(num_examples_total)

        # Create a list of weights, each multiplied by the related number of examples
        # new_weights = [
        #     [layer * result.num_examples for layer in parameters_to_ndarrays(result.parameters)] for _, result in results
        # ]
        # new_weights = [
        #     parameters_to_ndarrays(result.parameters) for _, result in results
        # ]

        weighted_gradients = []
        total_weight = 0.0
        for client, (parameters, num_examples) in results:
            client_weights = fl.common.parameters_to_weights(parameters)
            gradients = [c - g for c, g in zip(client_weights, self.current_weights)]
            weighted_gradients.append((num_examples, gradients))
            total_weight += num_examples    

        # print(self.current_weights, f"\n_____________________SEPARATOR_______________\n", new_weights)
        # print(type(self.current_weights), type(new_weights))

        # Accessing gradients
        # curr_gradients: NDArrays = [
        #     x - y for x, y in zip(new_weights, self.current_weights)
        # ]

        # weights_prime: NDArrays = [
        #     reduce(np.add, layer_updates) / num_examples_total
        #     for layer_updates in zip(*weighted_weights)
        # ]
        # Compute average gradients of each layer
        # gradients_prime: NDArrays = [
        #     reduce(np.add, layer_updates) / num_examples_total
        #     for layer_updates in zip(*we)
        # ]

        avg_gradient = [np.zeros_like(w) for w in self.current_weights]
        for num_examples, gradients in weighted_gradients:
            for i, grad in enumerate(gradients):
                avg_gradient[i] += (num_examples / total_weight) * grad

        return avg_gradient


    def aggregate_fit(self, server_round, results, failures):
        fedavg_gradients_aggregated = self.custom_aggregate(results=results)

        if fedavg_gradients_aggregated is None:
            return None, {}

        fedavg_gradients_converted = parameters_to_ndarrays(fedavg_gradients_aggregated)

        new_weights = [
            x + y for x, y in zip(self.current_weights, fedavg_gradients_converted)
        ]

        self.current_weights = new_weights

        return ndarrays_to_parameters(self.current_weights), metrics_aggregated
