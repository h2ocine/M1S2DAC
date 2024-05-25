import numpy as np

class Module:
    def __init__(self):
        self._parameters = None
        self._gradient = None
        self._X = None

    def zero_grad(self):
        """
        Réinitialise le gradient à zéro.
        """
        if self._parameters is not None:
            self._gradient = {k: np.zeros_like(v) for k, v in self._parameters.items()}

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Effectue une passe avant (calcul des sorties).
        :param X: données d'entrée, forme (m, d)
        """
        raise NotImplementedError("La méthode forward doit être implémentée par les sous-classes.")

    def _save_data(self, X: np.ndarray):
        """
        Sauvegarde les données d'entrée pour une utilisation lors de la rétropropagation.
        :param X: données d'entrée, forme (m, d)
        """
        self._X = X

    def __call__(self, X: np.ndarray):
        return self.forward(X)

    def update_parameters(self, gradient_step: float = 1e-3):
        """
        Met à jour les paramètres en utilisant le gradient calculé.
        :param gradient_step: taux d'apprentissage
        """
        if self._parameters is not None and self._gradient is not None:
            for k in self._parameters.keys():
                self._parameters[k] -= gradient_step * self._gradient[k]

    def set_parameters(self, parameters):
        """
        Définit les paramètres du module.
        :param parameters: dictionnaire de paramètres
        """
        self._parameters = parameters

    def get_parameters(self):
        """
        Retourne les paramètres du module.
        """
        return self._parameters

    def backward_update_gradient(self, X: np.ndarray, delta: np.ndarray):
        """
        Met à jour le gradient des paramètres.
        :param X: données d'entrée, forme (m, d)
        :param delta: delta de la couche courante, forme (m, d')
        """
        raise NotImplementedError("La méthode backward_update_gradient doit être implémentée par les sous-classes.")

    def backward_delta(self, X: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """
        Calcule le delta pour la couche précédente.
        :param X: données d'entrée, forme (m, d)
        :param delta: delta de la couche courante, forme (m, d')
        :return: delta pour la couche précédente, forme (m, d)
        """
        raise NotImplementedError("La méthode backward_delta doit être implémentée par les sous-classes.")
