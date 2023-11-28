import ploting as plot
import nueralnet as NN
import numpy as np

if __name__ == "__main__":
    np.random.seed(5280)
    X = (np.random.rand(200, 2) * 100) - 50
    Y = (X[:, 0] + 3 * X[:, 1] - 2 > 0).astype(int)

    # part 1
    for style in ["incremental", "batch"]:
        for lrate in [
            0.5,
            0.25,
            0.1,
            1e-2,
            1e-3,
            1e-4,
        ]:
            print(f"style: {style}, lrate: {lrate}")
            model = NN.unitNN(2, 1)
            errors, planes = model.train(X, Y, style, lrate)
            print("plotting")
            plot.plot_error(errors, style, lrate)
            print("plotting again")
            plot.plot_decision_planes(X, Y, planes)
