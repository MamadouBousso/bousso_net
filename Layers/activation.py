"""Module pour la couche d'activation

    
"""
import numpy as np
from bousso_net2.tenseurs.tenseur import Tenseur
from bousso_net2.Layers.layers import F, Layer


def sig(z: Tenseur) -> Tenseur:
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """
    s = 1.0 / (1.0 + np.exp(-z))
    return s


def sig_prime(z: Tenseur) -> Tenseur:
    return sig(z) * (1 - sig(z))


def tan(x: Tenseur) -> Tenseur:
    return np.tanh(x)


def tan_prime(x: Tenseur) -> Tenseur:
    y = tan(x)
    return 1 - y**2


def relu(x: Tenseur) -> Tenseur:
    """
    Compute the relu of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- relu(x)
    """
    s = np.maximum(0, x)

    return s


def relu_prime(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


class Activation(Layer):
    def __init__(self, f: F, f_prime: F):
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tenseur):
        self.inputs = inputs
        return self.f(self.inputs)

    def backwards(self, grad: Tenseur):
        self.f_prime(self.inputs) * grad


class Activation_tan(Activation):
    def __init__(self, tan, tan_prime):
        super().__init__(tan, tan_prime)


class Activation_sig(Activation):
    def __init__(self, sig, sig_prime):
        super().__init__(sig, sig_prime)


class Activation_relu(Activation):
    def __init__(self, relu, relu_prime):
        super().__init__(relu, relu_prime)
