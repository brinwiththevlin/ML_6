import numpy as np


class unitNN:
    def __init__(self, in_nodes, out_nodes=1):
        self.weights = np.random.randn(in_nodes + 1, 1)

    def activate(self, data: np.ndarray) -> bool:
        """activation function
        Args:
            data (np.ndarray):
        Returns:
            bool: output of the NN
        """
        assert (
            len(self.weights) == data.shape[1] + 1
        ), "number of inputs should mach initialized size."
        data = np.c_[np.ones((len(data), 1)), data]

        bool_np = np.matmul(data, self.weights) > 0
        bool_np[bool_np == True] = 1
        bool_np[bool_np == False] = 0

        return bool_np

    def train(self, dataset: np.ndarray, style: str, epochs: int) -> None:
        """trains the wieghs and biases on a dataset

        Args:
            dataset (np.ndarray): dataset to train on, include the labels
            style (str): batch, or incremental
            epochs (int): the number of iterations to train

        """
        assert style in [
            "batch",
            "incremental",
        ], "training style must be batch or incremental."

        if style == "batch":
            featureset = dataset[:, :2]
            labels = dataset[:, 2]
            for _ in range(epochs):
                outputs = self.activate(featureset)
                error_gradients = (
                    2 / len(featureset) * featureset.T.dot(outputs - labels)
                )
                self.weights -= error_gradients

        else:
            featureset = dataset[:, :2]
            labels = dataset[:, 2]
            for _ in range(epochs):
                for i in range(len(featureset)):
                    xi = featureset[i : i + 1, :]
                    yi = labels[i : i + 1, :]
                    output = self.activate(xi)
                    error_gradient = xi.T.dot(output - yi)
                    self.weights -= error_gradient
