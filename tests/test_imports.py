def test_mnist_wrapper_exists():
    import importlib
    m = importlib.import_module('src.task1.mnist_wrapper')
    assert hasattr(m, 'MnistClassifier')

def test_pipeline_module():
    import importlib
    m = importlib.import_module('src.task2.pipeline')
    assert hasattr(m, 'main')
