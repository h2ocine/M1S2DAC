import numpy as np
from module import Module
from sequential import Sequential
from lineaire_1 import Linear
from nonlineaire_2 import TanH, Sigmoid

class AutoEncoder(Sequential):
    def __init__(self, encoder: Module, decoder: Module):
        """
        Initialise un autoencodeur avec un encodeur et un décodeur.
        :param encoder: module de l'encodeur
        :param decoder: module du décodeur
        """
        super().__init__(encoder, decoder)

    def get_encoder(self) -> Module:
        """
        Retourne l'encodeur.
        """
        return self._modules[0]

    def get_decoder(self) -> Module:
        """
        Retourne le décodeur.
        """
        return self._modules[1]

    def parameters_sharing(self):
        """
        Partage les paramètres entre l'encodeur et le décodeur (transposée des poids).
        """
        def _set_parameters(parameters, module):
            if isinstance(module, Linear):
                parameters_ = dict()
                parameters_['W'] = parameters['W'].T
                parameters_['b'] = module._parameters['b']
                module.set_parameters(parameters_)

            elif isinstance(module, Sequential):
                cpt = 0
                for parameters_ in parameters:
                    for i in range(cpt, len(module._modules)):
                        cpt += 1
                        if not isinstance(module._modules[i], Activation):
                            _set_parameters(parameters_, module._modules[i])
                            break

        encoder_parameters = self._modules[0].get_parameters()
        encoder_parameters = list(filter(None, encoder_parameters))
        _set_parameters(reversed(encoder_parameters), self._modules[1])

# # Exemples d'utilisation

# # Création d'un autoencodeur
# encoder = Sequential(
#     Linear(256, 100),
#     TanH(),
#     Linear(100, 10),
#     TanH()
# )

# decoder = Sequential(
#     Linear(10, 100),
#     TanH(),
#     Linear(100, 256),
#     Sigmoid()
# )

# autoencoder = AutoEncoder(encoder, decoder)

# # Exemple de passe avant
# X = np.random.randn(32, 256)  # Batch de 32 exemples, chaque exemple de dimension 256
# encoded = autoencoder.get_encoder().forward(X)
# decoded = autoencoder.get_decoder().forward(encoded)

# # Exemple de rétropropagation
# from loss import MSELoss

# loss = MSELoss()
# yhat = autoencoder.forward(X)
# cost = loss.forward(X, yhat)

# delta = loss.backward(X, yhat)
# autoencoder.zero_grad()
# autoencoder.backward_update_gradient(X, delta)
# autoencoder.update_parameters(gradient_step=1e-3)
