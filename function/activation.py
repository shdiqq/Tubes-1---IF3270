import numpy as np

def linear(x):
  """Fungsi aktivasi linear"""
  return (x)

def sigmoid(x):
  """Fungsi aktivasi sigmoid"""
  return (1 / (1 + np.exp(-x)))

def ReLU(x):
  """Fungsi aktivasi ReLU"""
  return (np.maximum(0, x))

def softmax(x):
  """Fungsi aktivasi softmax"""
  return (np.exp(x) / np.sum(np.exp(x)))