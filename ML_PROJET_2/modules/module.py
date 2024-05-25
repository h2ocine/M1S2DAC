import numpy as np

class Module(object):

    def __init__(self):
        self._parameters = None
        self._gradient   = None
        self._X          = None


    def zero_grad(self):
        '''
        set the gradient to 0
        '''
        pass


    def forward(self, X: np.ndarray) -> np.ndarray:
        '''
        Compute the forward pass
        :param X: input data
            shape (m, d)
        '''
        raise NotImplementedError


    def _save_data(self, X: np.ndarray):
        '''
        Save forward data
        :param X: input data
            shape (m, d)
        '''
        self._X = X


    def __call__(self, X: np.ndarray):
        return self.forward(X)


    def update_parameters(self, gradient_step: float=1e-3):
        '''
        Calculation of the update of the parameters according to the calculated gradient 
        and the step of gradient_step
        :param gradient_step : learning rate
        '''
        pass


    def set_parameters(self, parameters):
        '''
        Set module parameters
        :param parameters
        '''
        pass

    
    def get_parameters(self):
        '''
        Get module parameters
        '''
        return None


    def backward_update_gradient(self, X: np.ndarray, delta: np.ndarray):
        '''
        Update the gradient value
        :param X: input data
            shape (m, d)
        :param delta: current layer delta
            shape (m, d')
        '''
        pass


    def backward_delta(self, X: np.ndarray, delta: np.ndarray) -> np.ndarray:
        '''
        Calculate the derivative of the error
        :param X: input data
            shape (m, d)
        :param delta: current layer delta
            shape (m, d')
        :return previous layer delta
            shape (m, d)
        '''
        raise NotImplementedError


class Initialization(object):
    '''
    module weight initialization mode
    zero, one, random, uniform, xavier, lecun
    '''
    ZERO    = 0
    ONE     = 1
    RANDOM  = 2
    UNIFORM = 3
    XAVIER  = 4
    LECUN   = 5


class GradientDescentMode(object):
    '''
    Gradient Descent mode
    batch, mini_batch, stochastic
    '''
    BATCH      = 0
    MINI_BATCH = 1
    STOCHASTIC = 2



# --------------------------------------------------------------------------- #
# ------------------------- Fonction d'activations -------------------------- #
# --------------------------------------------------------------------------- #
class Activation(module.Module):
    def __init__(self):
        super().__init__()


    def _backward(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


    def backward_delta(self, X: np.ndarray, delta: np.ndarray) -> np.ndarray:
        return delta * self._backward(X)


class TanH(Activation):
    def __init__(self):
        super().__init__()


    def forward(self, X: np.ndarray) -> np.ndarray:
        self._save_data(X)
        return np.tanh(X)


    def _backward(self, X: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(X) ** 2


class Sigmoid(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._save_data(X)
        return 1 / (1 + np.exp(-X))

    def _backward(self, X: np.ndarray) -> np.ndarray:
        return np.exp(-X) / ((1 + np.exp(-X)) ** 2)


class ReLU(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._save_data(X)
        return np.maximum(X, 0)

    def _backward(self, X: np.ndarray) -> np.ndarray:
        return np.where(X > 0, 1, 0)


class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._save_data(X)
        return np.maximum(X, self.alpha*X)

    def _backward(self, X: np.ndarray) -> np.ndarray:
        return np.where(X > 0, 1, self.alpha)


class ELU(Activation):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._save_data(X)
        return np.where(X > 0, X, self.alpha * (np.exp(X) - 1))

    def _backward(self, X: np.ndarray) -> np.ndarray:
        return np.where(X > 0, 1, self.alpha * np.exp(X))


class Softmax(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._save_data(X)
        return softmax(X)

    def _backward(self, X: np.ndarray) -> np.ndarray:
        s = self.forward(X)
        return s * (1 - s)
        


