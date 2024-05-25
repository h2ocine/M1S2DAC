import numpy as np

def softmax(X):
    """
    Calcule la fonction softmax.
    :param X: données d'entrée
    :return: données normalisées par softmax
    """
    exp_X = np.exp(X - np.max(X, axis=-1, keepdims=True))
    return exp_X / np.sum(exp_X, axis=-1, keepdims=True)

class Loss:
    def _assert_shape(self, y: np.ndarray, yhat: np.ndarray):
        """
        Vérifie que les dimensions de y et yhat sont compatibles.
        :param y: vérité terrain
        :param yhat: prédiction
        """
        assert y.shape == yhat.shape, 'La vérité terrain et les matrices de prédiction doivent avoir la même taille'

    def __call__(self, y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        return self.forward(y, yhat)

    def forward(self, y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        pass

    def backward(self, y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        pass

class MSELoss(Loss):
    """
    MSE Loss
    """
    def forward(self, y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        """
        Passe avant pour la MSE.
        :param y: vérité terrain
        :param yhat: prédiction
        :return: MSE
        """
        self._assert_shape(y, yhat)
        return (y - yhat)**2

    def backward(self, y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        """
        Passe arrière pour la MSE.
        :param y: vérité terrain
        :param yhat: prédiction
        :return: gradient de la MSE
        """
        self._assert_shape(y, yhat)
        return -2 * (y - yhat)

class BCELoss(Loss):
    """
    Binary Cross Entropy Loss
    """
    def forward(self, y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        """
        Passe avant pour la BCE.
        :param y: vérité terrain
        :param yhat: prédiction
        :return: BCE Loss
        """
        self._assert_shape(y, yhat)
        yhat = np.clip(yhat, 1e-12, 1 - 1e-12)
        return - (y * np.log(yhat) + (1 - y) * np.log(1 - yhat))

    def backward(self, y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        """
        Passe arrière pour la BCE.
        :param y: vérité terrain
        :param yhat: prédiction
        :return: gradient de la BCE
        """
        self._assert_shape(y, yhat)
        yhat = np.clip(yhat, 1e-12, 1 - 1e-12)
        return (-y / yhat) + ((1 - y) / (1 - yhat))

class CELoss(Loss):
    """
    Cross Entropy Loss avec Softmax
    """
    def forward(self, y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        """
        Passe avant pour la CE.
        :param y: vérité terrain
        :param yhat: prédiction
        :return: CE Loss
        """
        self._assert_shape(y, yhat)
        yhat = softmax(yhat)
        return -np.sum(y * np.log(yhat), axis=1)

    def backward(self, y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        """
        Passe arrière pour la CE.
        :param y: vérité terrain
        :param yhat: prédiction
        :return: gradient de la CE
        """
        self._assert_shape(y, yhat)
        return softmax(yhat) - y

class CCELoss(Loss):
    """
    Categorical Cross Entropy Loss
    """
    def forward(self, y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        """
        Passe avant pour la CCE.
        :param y: vérité terrain
        :param yhat: prédiction
        :return: CCE Loss
        """
        self._assert_shape(y, yhat)
        yhat = np.clip(yhat, 1e-12, 1 - 1e-12)
        return -np.sum(y * np.log(yhat), axis=1)

    def backward(self, y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        """
        Passe arrière pour la CCE.
        :param y: vérité terrain
        :param yhat: prédiction
        :return: gradient de la CCE
        """
        self._assert_shape(y, yhat)
        yhat = np.clip(yhat, 1e-12, 1 - 1e-12)
        return -y / yhat

class HingeLoss(Loss):
    """
    Hinge Loss
    """
    def __init__(self, alpha: float = 1):
        self.alpha = alpha

    def forward(self, y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        """
        Passe avant pour la Hinge Loss.
        :param y: vérité terrain
        :param yhat: prédiction
        :return: Hinge Loss
        """
        self._assert_shape(y, yhat)
        return np.maximum(0, self.alpha - y * yhat)

    def backward(self, y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        """
        Passe arrière pour la Hinge Loss.
        :param y: vérité terrain
        :param yhat: prédiction
        :return: gradient de la Hinge Loss
        """
        self._assert_shape(y, yhat)
        return np.where(y * yhat < self.alpha, -y, 0)
