import numpy as np
from module import Module
from lineaire_1 import Linear
from nonlineaire_2 import Activation

class Sequential(Module):
    def __init__(self, *modules):
        """
        Initialise une séquence de modules.
        :param modules: liste de modules à chaîner
        """
        super().__init__()
        self._modules = list(modules)
        self._last_linear_module_output = None
        for module in modules:
            self.add_module(module)

    def add_module(self, module: Module):
        """
        Ajoute un module à la séquence.
        :param module: module à ajouter
        """
        assert isinstance(module, Module), 'Le module doit être une instance de la classe Module'
        if self._last_linear_module_output is not None:
            if isinstance(module, Linear):
                assert self._last_linear_module_output == module.input, \
                    ('La taille de sortie du dernier module linéaire doit être égale à la taille d\'entrée du nouveau module')
        if isinstance(module, Linear):
            self._last_linear_module_output = module.output
        elif isinstance(module, Sequential):
            self._last_linear_module_output = module._last_linear_module_output
        self._modules.append(module)

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._save_data(X)
        for module in self._modules:
            X = module.forward(X)
        return X

    def zero_grad(self):
        for module in self._modules:
            module.zero_grad()

    def backward_update_gradient(self, X: np.ndarray, delta: np.ndarray):
        for module in reversed(self._modules):
            module.backward_update_gradient(module._X, delta)
            delta = module.backward_delta(module._X, delta)

    def backward_delta(self, X: np.ndarray, delta: np.ndarray) -> np.ndarray:
        for module in reversed(self._modules):
            delta = module.backward_delta(module._X, delta)
        return delta

    def update_parameters(self, gradient_step: float = 0.001):
        for module in self._modules:
            module.update_parameters(gradient_step)

    def set_parameters(self, parameters):
        assert len(parameters) == len(self._modules), 'Le nombre de paramètres doit correspondre au nombre de modules'
        for i in range(len(parameters)):
            self._modules[i].set_parameters(parameters[i])

    def get_parameters(self):
        return [module.get_parameters() for module in self._modules]
