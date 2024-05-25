import numpy as np
from module import Module
from nonlineaire_2 import Softmax
from loss import CELoss, CCELoss

class SoftmaxLayer(Module):
    def __init__(self):
        """
        Initialise une couche avec activation Softmax.
        """
        super().__init__()
        self.softmax = Softmax()

    def forward(self, X: np.ndarray) -> np.ndarray:
        return self.softmax.forward(X)

    def backward_delta(self, X: np.ndarray, delta: np.ndarray) -> np.ndarray:
        return self.softmax.backward_delta(X, delta)

# Exemples d'utilisation

# Création d'un réseau multiclasses avec la fonction d'activation softmax en sortie
# from linear import Linear
# from sequential import Sequential

# # Définition du réseau
# net = Sequential(
#     Linear(64, 128),
#     SoftmaxLayer()
# )

# # Fonction de coût Cross Entropy
# loss = CELoss()

# # Exemple de passe avant et calcul du coût
# X = np.random.randn(32, 64)  # Batch de 32 exemples, chaque exemple de dimension 64
# y = np.zeros((32, 128))  # Supervision, one-hot encoded
# y[np.arange(32), np.random.randint(0, 128, 32)] = 1  # Exemples de cibles aléatoires

# yhat = net.forward(X)
# cost = loss.forward(y, yhat)

# # Exemple de rétropropagation
# delta = loss.backward(y, yhat)
# net.zero_grad()
# net.backward_update_gradient(X, delta)
# net.update_parameters(gradient_step=1e-3)
