import numpy as np
import module
import nonlineaire_2

class SoftmaxLayer(module.Module):
    def __init__(self):
        """
        Initialise une couche avec activation Softmax.
        """
        super().__init__()
        self.softmax = nonlineaire_2.Softmax()

    def forward(self, X: np.ndarray) -> np.ndarray:
        return self.softmax.forward(X)

    def backward_delta(self, X: np.ndarray, delta: np.ndarray) -> np.ndarray:
        return self.softmax.backward_delta(X, delta)
    