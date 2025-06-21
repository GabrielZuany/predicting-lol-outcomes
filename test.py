import tensorflow as tf
import torch

def test_cpu_gpu():
    # Test TensorFlow
    print("TensorFlow version:", tf.__version__)
    print("TensorFlow GPU available:", tf.config.list_physical_devices('GPU'))
    # Test PyTorch
    print("PyTorch version:", torch.__version__)
    print("PyTorch GPU available:", torch.cuda.is_available())

if __name__ == "__main__":
    test_cpu_gpu()