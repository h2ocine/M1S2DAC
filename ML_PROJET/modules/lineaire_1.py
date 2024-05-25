import numpy as np
from module import Module

class Initialization:
    """
    Modes d'initialisation des poids du module.
    zero, one, random, uniform, xavier, lecun
    """
    ZERO = 0
    ONE = 1
    RANDOM = 2
    UNIFORM = 3
    XAVIER = 4
    LECUN = 5

class GradientDescentMode:
    """
    Modes de descente de gradient.
    batch, mini_batch, stochastic
    """
    BATCH = 0
    MINI_BATCH = 1
    STOCHASTIC = 2

class Linear(Module):
    def __init__(self, input: int, output: int, bias: bool = True, initialization: int = Initialization.LECUN):
        """
        Initialise une couche linéaire.
        :param input: taille de chaque échantillon d'entrée
        :param output: taille de chaque échantillon de sortie
        :param bias: si False, la couche n'apprendra pas de biais. Par défaut: True
        :param initialization: mode d'initialisation
        """
        super().__init__()
        self.input = input
        self.output = output
        self.bias = bias
        self._parameters = dict()
        self._gradient = dict()
        self._init_parameters(initialization)
        self.zero_grad()

    def _init_parameters(self, initialization: int):
        """
        Initialise les paramètres du module selon le mode spécifié.
        :param initialization: mode d'initialisation
        """
        input, output = self.input, self.output
        shape = (input, output)

        if initialization == Initialization.ZERO:
            self._parameters['W'] = np.zeros(shape)
        elif initialization == Initialization.ONE:
            self._parameters['W'] = np.ones(shape)
        elif initialization == Initialization.RANDOM:
            self._parameters['W'] = np.random.random(shape)
        elif initialization == Initialization.UNIFORM:
            self._parameters['W'] = np.random.uniform(-np.sqrt(1/input), np.sqrt(1/input), shape)
        elif initialization == Initialization.XAVIER:
            self._parameters['W'] = np.random.randn(*shape) * np.sqrt(2 / (input + output))
        elif initialization == Initialization.LECUN:
            self._parameters['W'] = np.random.randn(*shape) * np.sqrt(1 / input)
        else:
            raise ValueError("Unknown initialization method")

        if self.bias:
            self._parameters['b'] = np.zeros(output)

    # def forward(self, X: np.ndarray) -> np.ndarray:
    #     """
    #     Effectue une passe avant (calcul des sorties).
    #     :param X: données d'entrée, forme (batch, input)
    #     :return: sorties, forme (batch, output)
    #     """
    #     self._save_data(X)
    #     output = X.dot(self._parameters['W'])
    #     if self.bias:
    #         output += self._parameters['b']
    #     return output
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        self._save_data(X)
        print(f"Forward pass - shape of input X: {X.shape}")
        output = X.dot(self._parameters['W'])
        if self.bias:
            output += self._parameters['b']
        print(f"Forward pass - shape of output: {output.shape}")
        return output


    def backward_update_gradient(self, X: np.ndarray, delta: np.ndarray):
        """
        Met à jour le gradient des paramètres.
        :param X: données d'entrée, forme (batch, input)
        :param delta: delta de la couche courante, forme (batch, output)
        """
        self._gradient['W'] = X.T.dot(delta)
        if self.bias:
            self._gradient['b'] = np.sum(delta, axis=0)

    def backward_delta(self, X: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """
        Calcule le delta pour la couche précédente.
        :param X: données d'entrée, forme (batch, input)
        :param delta: delta de la couche courante, forme (batch, output)
        :return: delta pour la couche précédente, forme (batch, input)
        """
        return delta.dot(self._parameters['W'].T)
