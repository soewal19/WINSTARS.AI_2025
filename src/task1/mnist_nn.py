from .mnist_interface import MnistClassifierInterface
class MnistNN(MnistClassifierInterface):
    def __init__(self, input_dim=784):
        self.model = None
        self.input_dim = input_dim
    def _build(self):
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        inputs = keras.Input(shape=(self.input_dim,))
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(10, activation='softmax')(x)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    def train(self, X, y, **kwargs):
        model = self._build()
        model.fit(X, y, epochs=kwargs.get('epochs',3), batch_size=kwargs.get('batch_size',64), verbose=kwargs.get('verbose',1))
        self.model = model
    def predict(self, X):
        import numpy as np
        preds = self.model.predict(X)
        return np.argmax(preds, axis=1)
    def save(self, path):
        self.model.save(path)
    def load(self, path):
        import tensorflow as tf
        self.model = tf.keras.models.load_model(path)
