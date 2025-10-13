# Task 1: MNIST Classification with OOP

This task implements three different classification algorithms for the MNIST dataset using Object-Oriented Programming principles.

## Models Implemented

1. **Random Forest** - Uses scikit-learn's RandomForestClassifier
2. **Feed-Forward Neural Network** - Uses TensorFlow/Keras dense layers
3. **Convolutional Neural Network** - Uses TensorFlow/Keras CNN layers

## Interface

All models implement the `MnistClassifierInterface` which defines the following methods:
- `train(X, y, **kwargs)` - Train the model
- `predict(X)` - Make predictions
- `save(path)` - Save the model to disk
- `load(path)` - Load the model from disk

## Wrapper Class

The `MnistClassifier` class acts as a factory that instantiates the appropriate model based on the algorithm parameter:
- 'rf' for Random Forest
- 'nn' for Neural Network
- 'cnn' for Convolutional Neural Network

## Usage

```bash
python train_mnist.py --algo {rf|nn|cnn} --epochs <int> --limit <int>
```

## Examples

```bash
# Train CNN with 3 epochs and limited dataset
python train_mnist.py --algo cnn --epochs 3 --limit 2000

# Train Random Forest
python train_mnist.py --algo rf --limit 5000
```