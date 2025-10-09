from .mnist_interface import MnistClassifierInterface
class MnistRF(MnistClassifierInterface):
    def __init__(self, **kwargs):
        self.model = None
    def train(self, X, y, **kwargs):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=kwargs.get('n_estimators',100))
        self.model.fit(X, y)
    def predict(self, X):
        return self.model.predict(X)
    def save(self, path):
        import joblib
        joblib.dump(self.model, path)
    def load(self, path):
        import joblib
        self.model = joblib.load(path)
