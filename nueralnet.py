import numpy as np
from typing import Tuple, List


class unitNN:
    def __init__(self, in_nodes, out_nodes=1):
        self.weights = np.random.randn(in_nodes + 1)
        self.reset_weights = self.weights

    def activate(self, data: np.ndarray) -> bool:
        """activation function
        Args:
            data (np.ndarray):
        Returns:
            bool: output of the NN
        """
        assert (
            len(self.weights) == data.shape[1]
        ), "number of inputs should mach initialized size."

        bool_np = (np.matmul(data, self.weights) > 0).flatten().astype(int)

        return bool_np

    def reset(self):
        self.weights = self.reset_weights

    def train(
        self, X: np.ndarray, Y: np.ndarray, style: str, lrate: float
    ) -> Tuple[List[int], List[Tuple[float, float]]]:
        """trains the wieghs and biases on a dataset

        Args:
            X (np.ndarray): dataset to train on
            Y (np.ndarray): the labels
            style (str): batch, or incremental
            lrate (float): learning rate
        Returns:
            List[int]: errors
            List[(float, float)]
        """
        assert style in [
            "batch",
            "incremental",
        ], "training style must be batch or incremental."

        self.reset()
        errors = []
        planes = []
        if style == "batch":
            featureset = np.c_[np.ones((len(X), 1)), X]
            labels = Y
            for i in range(1, 101):
                outputs = self.activate(featureset)
                error_gradients = (
                    2 / len(featureset) * featureset.T.dot(outputs - labels)
                )
                self.weights -= error_gradients * lrate
                errors.append(sum(np.abs(outputs - labels)))
                if i in [5, 10, 50, 100]:
                    planes.append(self.weights)

        else:
            featureset = np.c_[np.ones((len(X), 1)), X]
            labels = Y
            for _ in range(1, 101):
                step_error = 0
                for i in range(len(featureset)):
                    xi = featureset[i : i + 1, :]
                    yi = labels[i : i + 1]
                    output = self.activate(xi)
                    error_gradient = xi.T.dot(output - yi)
                    self.weights -= error_gradient * lrate
                    step_error += sum(np.abs(output - yi))
                    if i in [5, 10, 50, 100]:
                        planes.append(self.weights)
                errors.append(step_error)

        return errors, planes
