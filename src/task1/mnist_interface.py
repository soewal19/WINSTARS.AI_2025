from abc import ABC, abstractmethod

class MnistClassifierInterface(ABC):
    @abstractmethod
    def train(self, X, y, **kwargs):
        pass
    @abstractmethod
    def predict(self, X):
        pass
    @abstractmethod
    def save(self, path):
        pass
    @abstractmethod
    def load(self, path):
        pass
