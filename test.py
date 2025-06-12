import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from sklearn.datasets import load_iris
from transformers import pipeline

def test_cpu_gpu():
    # Test TensorFlow
    print("TensorFlow version:", tf.__version__)
    print("TensorFlow GPU available:", tf.config.list_physical_devices('GPU'))
    # Test PyTorch
    print("PyTorch version:", torch.__version__)
    print("PyTorch GPU available:", torch.cuda.is_available())
    # Test basic ML
    iris = load_iris()
    print("Sklearn dataset loaded successfully")
    # Test transformers
    classifier = pipeline('sentiment-analysis')
    result = classifier("Testing my AI environment!")
    print("Transformers test:", result)
if __name__ == "__main__":
    test_cpu_gpu()