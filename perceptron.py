import numpy as np
import random
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self):
        self.w = np.zeros((3, 1))

    def gen_targetfunc(self, N):
        # Get the target function
        w = np.array([1, random.uniform(-1, 1),
                      random.uniform(-1, 1)])[np.newaxis]
        print w.T

        # Generate training data from it
        X = np.zeros((N, 3))
        for i in range(0, X.shape[0]):
            X[i, :] = (1, random.uniform(-1, 1), random.uniform(-1, 1))
        print X[0, :][np.newaxis]
        self.data = X
        self.func = w
        y = np.inner(w, X)
        self.target = y
        self.target[y > 0] = 1
        self.target[y < 0] = -1
        print self.target

    def display(self, X, y, w):
        # create a mesh to plot in
        h = .02
        x_min, x_max = X[:, 1].min() - 10, X[:, 1].max() + 10
        y_min, y_max = X[:, 2].min() - 10, X[:, 2].max() + 10
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        Z = np.inner(w, np.c_[np.ones(xx.ravel().shape),
                              xx.ravel(),
                              yy.ravel()])
        Z[Z >= 0] = 1
        Z[Z < 0] = -1
        Z = Z.reshape(xx.shape)
        print 'Z', Z
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
        pass

    def test(self):
        # TODO
        pass

percept = Perceptron()
percept.gen_targetfunc(10)
percept.display(percept.data, percept.target, percept.func)
