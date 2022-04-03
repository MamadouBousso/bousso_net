"""_summary_
Une fonction perte permet de mesurer la qualité des predictions du modele
Il en existe plusieurs types: MSE, MAE, Huber, 
"""
import numpy as np  # type: ignore
from bousso_net2.tenseurs.tenseur import Tenseur


class Loss:
    def __init__(self) -> None:
        raise NotImplementedError

    def loss(self, predicted: Tenseur, actual: Tenseur) -> float:
        """_summary_
        Calcul de la fonction perte
        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    def gradLoss(self, predicted: Tenseur, actual: Tenseur) -> Tenseur:
        raise NotImplementedError


class MSE(Loss):
    """Cette classe decrit le loss et son gradient pour le MSE

    Args:
        Loss (None): fonction perte pour le MSE
    """

    def __init__(self):
        raise NotImplementedError

    def loss(self, predicted: Tenseur, actual: Tenseur) -> float:
        return np.sum((predicted - actual) ** 2)

    def gradLoss(self, predicted: Tenseur, actual: Tenseur) -> Tenseur:
        return 2 * (predicted - actual)


class LogisticLoss(Loss):
    """Fonction perte pour la regression logistique

    Args:
        Loss (): Fonction perte pour la regression logistique
    """

    def __init__(self):
        raise NotImplementedError

    def loss(self, predicted: Tenseur, actual: Tenseur) -> float:
        """
        Implement the loss function

        Arguments:
        predicted -- post-activation, output of forward propagation
        actual -- "true" labels vector, same shape as a3

        Returns:
        loss_val - value of the loss function
        """
        m = actual.shape[1]
        loss_val = (
            1.0
            / m
            * np.nansum(
                -np.multiply(actual, np.log(predicted))
                - np.multiply(1 - actual, np.log(1 - predicted))
            )
        )
        return loss_val

    def gradLoss(self, predicted: Tenseur, actual: Tenseur) -> Tenseur:
        """
        Implement the gradient of the loss function

        Arguments:
        predicted -- post-activation, output of forward propagation
        actual -- "true" labels vector, same shape as predicted

        Returns:
        value of the gradient of the loss function
        """
        m = actual.shape[1]
        return (
            1.0
            / m
            * (
                -np.multiply(actual, 1.0 / predicted)
                + np.multiply(1 - actual, 1.0 / (1 - predicted))
            )
        )
