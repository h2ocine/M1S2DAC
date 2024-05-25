import numpy as np
import module
from lineaire_1 import Linear
from nonlineaire_2 import TanH, Sigmoid
import encapsulage_3
import activation
import lineaire_1

class AutoEncoder(encapsulage_3.Sequential):

    def __init__(self, encoder: module.Module, decoder: module.Module):
        super().__init__(encoder, decoder)
        

    def get_encoder(self) -> module.Module:
        return self._modules[0]


    def get_decoder(self) -> module.Module:
        return self._modules[1]


    def parameters_sharing(self):
        def _set_parameters(parameters, module):
            if isinstance(module, lineaire_1.Linear):
                parameters_ = dict()
                parameters_['W'] = parameters['W'].T
                parameters_['b'] = module._parameters['b']
                module.set_parameters(parameters_)

            elif isinstance(module, encapsulage_3.Sequential):
                cpt = 0
                for parameters_ in parameters:
                    for i in range(cpt, len(module._modules)):
                        cpt += 1
                        if not isinstance(module._modules[i], activation.Activation):
                            _set_parameters(parameters_, module._modules[i])
                            break

        encoder_parameters = self._modules[0].get_parameters()
        encoder_parameters = list(filter(None, encoder_parameters))
        _set_parameters(reversed(encoder_parameters), self._modules[1])