"""_summary_
Ce module permet de definir les couches:
Il existe 2 types de couches en deep learning:
Les couches lineaires et les couches d'activation
Dans une couche on fait du forward et du backward
"""
import numpy as np
from typing import Dict, Callable

from bousso_net2.Layers.initialize import (
    Initializer,
    HeInitializer,
    RandInitializer,
    ZerosInitializer,
)
from bousso_net2.tenseurs.tenseur import Tenseur

F: Callable[[Tenseur], Tenseur]


class Layer:
    def __init__(self, input_size: int, output_size: int) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.params: Dict[str, Tenseur] = {}
        self.grad: Dict[str, Tenseur] = {}

    def forward(self, inputs: Tenseur) -> Tenseur:
        raise NotImplementedError

    def backward(self, grad: Tenseur) -> Tenseur:
        raise NotImplementedError
