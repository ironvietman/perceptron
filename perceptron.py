import numpy as np
import matplotlib.pyplot as plt
import random

class Perceptron:
    def __init__(self):
        self.w = np.zeros((3, 1))
        self.data = None
        self.target = None
        self.func = None

    def gen_targetfunc(self, N):
        # Get the target function
        w = np.array([1, np.random.uniform(-.5, .5),
                      np.random.uniform(-5, .5)])[np.newaxis]
        # print w.T

        # Generate training data from it
        X = np.random.uniform(-1, 1, (N, 2))
        X = np.hstack((np.ones((N, 1)), X))

        self.data = X
        self.func = w
        y = np.inner(w, X)
        self.target = y
        self.target[y > 0] = 1
        self.target[y < 0] = -1
        # print self.target

    def display(self, X, y, w):
        # create a mesh to plot in
        h = .02
        x_min, x_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        y_min, y_max = X[:, 2].min() - 1, X[:, 2].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        Z = np.inner(w, np.c_[np.ones(xx.ravel().shape),
                              xx.ravel(),
                              yy.ravel()])
        Z[Z >= 0] = 1
        Z[Z < 0] = -1
        Z = Z.reshape(xx.shape)
        # print 'Z', Z
        plt.figure()
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
        # Plot also the training points
        plt.scatter(X[:, 1], X[:, 2], c=y, cmap=plt.cm.Paired)
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal width')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.show()

    def train(self):
        # TODO find the w's
        w = np.zeros((1, 3))
        X = self.data
        iterations = 0
        target = self.target
        y = target * -1  # all misclassified

        # interation
        while(True):
            # check for misclassifications
            misclassified = []
            for i in range(0, X.shape[0]):
                if(target[:, i]*y[:, i] < 0):
                    misclassified.append(i)

            # print len(misclassified)
            # nothing misclassified
            if(len(misclassified) == 0 or iterations > 1000):
                break

            # Adujust the weights for one point
            sample_id = random.choice(misclassified)
            sample = X[sample_id, :]
            adjust = target[:, sample_id]
            y[:, sample_id] = adjust

            # print adjust, y
            # print np.array([1, adjust, adjust])*sample
            # print sample
            w = w + adjust*sample  # adjust weights
            # percept.display(percept.data, percept.target, w)
            iterations += 1

        self.w = w
        return iterations

    def test(self, sample):
        y = cmp(np.inner(self.w, sample[np.newaxis]), 0)
        return y
        pass

percept = Perceptron()
percept.gen_targetfunc(100)
percept.display(percept.data, percept.target, percept.func)
plt.show()
itera = 0
RUNS = 100
for i in range(0, RUNS):
    # print i
    itera += percept.train()
print 'average iterations: ', itera/RUNS
#percept.display(percept.data, percept.target, percept.w)
