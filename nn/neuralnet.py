"""Construction d'un reseau de neurones avec des couches
"""

from bousso_net2.Layers.layers import Layer

from typing import Sequence

class NeuralNet:
    def __init__(self,couches: Sequence[Layer]):
        self.reseau  = couches
    
    def forward(self,inputs):
        for couche in self.reseau:
            inputs = couche.forward(inputs)
        return inputs
    
    def backwards(self, grad):
        for couche in self.reseau.__reversed__():
            grad = couche.backward(grad)
        return grad