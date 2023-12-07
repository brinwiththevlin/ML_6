import ploting as plot
import nueralnet as NN
import numpy as np
import time

if __name__ == "__main__":
    np.random.seed(5280)
    X = (np.random.rand(200, 2) * 100) - 50
    Y = (X[:, 0] + 3 * X[:, 1] - 2 > 0).astype(int)

    '''
    # part 1 and 2a
    for style in ["incremental", "batch", "batch-decaying-lrate"]:
        start = time.time()
        for lrate in [0.5, 0.25, 0.1, 1e-2, 1e-3, 1e-4]:
            print(f"style: {style}, lrate: {lrate}")
            model = NN.unitNN(2, 1)
            errors, planes = model.train(X, Y, style, lrate)
            print("plotting error")
            plot.plot_error(errors, style, lrate)
            print("plotting boundaries")
            plot.plot_decision_planes(X, Y, planes, style, lrate)
        end = time.time()
        print(f"total time spend on {style}: {end - start}")

'''

#Part 2b
lrate = .3
model = NN.unitNN(2, 1)
print("Adaptive learning rates, starting at .3")
model = NN.unitNN(2, 1)
errors, planes = model.train_adapt_rates(X, Y, lrate)
plot.plot_error(errors, "adaptive-rates", lrate)
plot.plot_decision_planes(X, Y, planes, "adaptive-rates", lrate)

'''
DISCUSSION 

The pics folder contains all the plots for all the learning rates and styles. 
When comparing the error plots for incremental vs batch training, you can 
see that with incremental, the error seems to jump around a lot. This is different
from batch training, as the error decreases rapidly and stays low. The end result is 
pretty similar, though. This pattern is consistent with the different learning 
rates. Since the incremental updates the network after EACH example, it makes sense 
that it fluctuates more. 

The decaying learning rate error plot has a similar phenomenon as the regular batch training,
probably due to it being trained as a batch as well. HOwever, a key difference
is that for learning rates less than 1, the error stayed the exact same throughout. 
It is just one straight line. This is most likely due to the learning rate already 
starting very small, so decreasing it just does nothing to the network.

'''