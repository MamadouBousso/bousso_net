import numpy as np


class Initializer:
    def __init__(self):
        raise NotImplementedError

    def initialize(self, input_size, output_size):
        raise NotImplementedError


class ZerosInitializer(Initializer):
    def __init__(self):
        raise NotImplementedError

    def initialize(self, input_size, output_size):
        return np.zeros((input_size, output_size))


class RandInitializer(Initializer):
    def __init__(self):
        np.random.seed(3)

    def initialize(self, input_size, output_size):
        return np.random.randn(input_size, output_size) * 10


class HeInitializer(Initializer):
    def __init__(self):
        np.random.seed(3)

    def initialize(self, input_size, output_size):
        return np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
