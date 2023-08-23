import numpy as np


# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

class Perceptron(object):
    """Perceptron classifier.

    Parameters
    --------------
    eta : float
      Learning rate (Between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.

    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    errors_ : list
      Number of miss classifications (updates) in each epoch.

    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.errors_ = []
        self.w_ = None
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, x, y):
        """Fit training data.

        Parameters
        --------------
        x : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples
            and n_features is the numbers of features
        y : array-like,  shape = [n_samples]
          Target values

        Returns
        --------------
        self : object

        """

        rgen = np.random.RandomState(self.random_state)
        # rgene = np.random.Generator(self)
        # ~numpy.random.Generator.normal
        # ~numpy.random.Generator

        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + x.shape[1])
        # self.w_ = rgene.random().Generator.normal

        for _ in range(self.n_iter):
            errors = 0
            # Producto escalar de x y y
            for xi, target in zip(x, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, x):
        """Calculate net input"""
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def predict(self, x):
        """Return class label after unit step"""
        return np.where(self.net_input(x) >= 0.0, 1, -1)

    v1 = np.array([1, 2, 3])
    v2 = 0.5 * v1
    print(np.arccos(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
