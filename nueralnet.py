import numpy as np
class unitNN:
    def __init__(self, in_nodes, out_nodes=1):
        self.weights = np.zeros(in_nodes)
        self.bias = 0
        
    def activate(self, data: np.ndarray) -> bool:
        """activation function
        Args:
            data (np.ndarray): 
        Returns:
            bool: output of the NN
        """
        assert len(self.weights) == data.shape[1] , "number of inputs should mach initialized size."
        
        return np.matmul(data, self.weights) + self.bias > 0
    
    def train(self, dataset: np.ndarray, style: str, epochs: int) -> None:
        """trains the wieghs and biases on a dataset

        Args:
            dataset (np.ndarray): dataset to train on, include the labels
            style (str): batch, or incremental
            epochs (int): the number of iterations to train
            
        """
        assert style in ['batch','incremental'], "training style must be batch or incremental."
        
        if style == 'batch':
            featureset = dataset[:, :2]
            labels = dataset[:, 2]
            for _ in range(epochs):
                
                outputs = self.activate(featureset)
                total_error = np.sum(labels != outputs)
                    
        