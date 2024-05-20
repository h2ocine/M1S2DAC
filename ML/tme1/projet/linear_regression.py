import projet_etu 


import numpy as np

class Linear(projet_etu.Module):
    """
    Classe pour la couche linéaire.
    """
    def __init__(self,  input, output):
        self._input = input
        self._output = output
        self._parameters = 2 * ( np.random.rand(self._input, self._output) - 0.5 ) #matrice de poids (initialisation aléatoire centrée en 0)
        self._biais = np.random.random((1, self._output)) - 0.5
        self.zero_grad() #on initialise en mettant les gradients à 0

    def forward(self, X):
        """"
        Permet de calculer les sorties du module pour les entrées passées en paramètre. 
        X : matrice des entrées (taille batch x input)
        Return : sorties du module (taille batch x output)
        """
        assert X.shape[1] == self._input
        return np.dot(X, self._parameters) + self._biais
    
    def zero_grad(self):
        """
        Permet de réinitialiser le gradient à 0.
        """
        self._gradient=np.zeros((self._input, self._output))
        self._biais_grad = np.zeros((1, self._output))
    
    def update_parameters(self, gradient_step=0.001):
        """
        Permet de mettre à jour les paramètres du module selon le gradient accumulé jusqu'à son appel avec un pas gradient_step
        gradient_step : pas du gradient
        """
        self._parameters -= gradient_step * self._gradient
        self._biais -= gradient_step * self._biais_grad
    
    def backward_update_gradient(self, input, delta):
        """
        Permet de calculer le gradient du coût par rapport aux paramètres et l’additionner à la variable _gradient en fonction de l’entrée input et des δ de la couche suivante delta.
        input: entrée du module
        delta: delta de la couche suivante
        """
        self._gradient += np.dot(input.T, delta)
        self._biais_grad += np.sum(delta, axis=0)

    def backward_delta(self, input, delta):
        """
        Permet de calculer le gradient du coût par rapport aux entrées en fonction de l’entrée input et des deltas de la couche suivante delta.
        input: entrée du module
        delta: delta de la couche suivante
        Return: delta de la couche actuelle
        """
        #assert input.shape[1] == self._input
        #assert delta.shape[1] == self._output
        return np.dot(delta, self._parameters.T)