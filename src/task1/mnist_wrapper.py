from .mnist_rf import MnistRF
from .mnist_nn import MnistNN
from .mnist_cnn import MnistCNN
class MnistClassifier:
    def __init__(self, algorithm='cnn', **kwargs):
        algorithm = algorithm.lower()
        if algorithm == 'rf':
            self.model = MnistRF(**kwargs)
        elif algorithm == 'nn':
            self.model = MnistNN(**kwargs)
        elif algorithm == 'cnn':
            self.model = MnistCNN(**kwargs)
        else:
            raise ValueError('Unknown algorithm')
    def train(self, X, y, **kwargs):
        return self.model.train(X, y, **kwargs)
    def predict(self, X):
        return self.model.predict(X)
    def save(self, path):
        return self.model.save(path)
    def load(self, path):
        return self.model.load(path)
