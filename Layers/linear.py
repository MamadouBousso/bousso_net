"""Linear layer class
"""
import numpy as np
from bousso_net2.tenseurs.tenseur import Tenseur
from bousso_net2.Layers.layers import Layer
from bousso_net2.Layers.initialize import (
    Initializer,
    HeInitializer,
    RandInitializer,
    ZerosInitializer,
)


class Linear(Layer):
    def __init__(
        self, input_size: int, output_size: int, initialize: Initializer, choix: str
    ) -> None:
        super().__init__(input_size, output_size)
        initializer = self.initialize(choix)

        self.params["b"] = np.zeros(output_size)
        self.params["W"] = initialize.initialize(input_size, output_size)

    def initialize(self, choix):
        """Initialisation des paramétres de la couche lineaire

        Args:
            choix (str): choix entre une initialisation avec zeros, random et He

        Returns:
            Objet : permettant le choix
        """
        if choix == "zeros":
            return ZerosInitializer()
        elif choix == "random":
            return RandInitializer()
        else:
            return HeInitializer()

    def forward(self, inputs: Tenseur) -> float:
        self.inputs = inputs
        return np.dot(self.params["W"], inputs) + self.params["b"]

    def backward(self, grad: Tenseur) -> Tenseur:
        self.grad["W"] = np.dot(self.inputs.T, grad)
        self.grad["b"] = np.sum(grad, axis=0)
        return np.dot(grad, self.params["W"].T)
