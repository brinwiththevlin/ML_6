import ploting as plot
import nueralnet as NN
import numpy as np

if __name__ == "__main__":
    np.random.seed(5280)
    X = (np.random.rand(200, 2) * 100) - 50
    Y = (X[:, 0] + 3 * X[:, 1] - 2 > 0).astype(int)

    model = NN.unitNN(2, 1)
    errors, planes = model.train(X, Y, "batch", 0.001)

    plot.plot_error(errors)
