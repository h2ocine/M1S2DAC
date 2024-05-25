import numpy as np
from module import Module
from lineaire_1 import Linear, GradientDescentMode
from nonlineaire_2 import Activation
import loss


class Sequential(Module):
    def __init__(self, *modules):
        self._modules = []
        self._last_linear_module_output = None
        for module in modules:
            self.add_module(module)


    def add_module(self, module: Module):
        assert isinstance(module, Module), 'The module object must be an instance of the Module class'
        
        if self._last_linear_module_output is not None:
            if isinstance(module, Linear):
                assert self._last_linear_module_output == module.input, ('The output size of the last linear ' + 
                'module must be equal to the input size of the new module')
        
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
        for module in reversed(self._modules):
            module.update_parameters(gradient_step)
    

    def set_parameters(self, parameters):
        assert len(parameters) == len(self._modules), 'The number of parameters and the number of modules must match'
        for i in range(len(parameters)):
            self._modules[i].set_parameters(parameters[i])


    def get_parameters(self):
        parameters = []
        for module in self._modules:
            parameters.append(module.get_parameters())
        return parameters
    



#----------------------------------------------------------------------------
#-------------------------OPTIMIZERS-----------------------------------------
#----------------------------------------------------------------------------



class Optim(object):

    def __init__(self, net: Module, loss: loss.Loss, eps: float):
        '''
        :param net: 
            A neural network implemented as a subclass of module.Module.
        :param loss: 
            A loss function implemented as a subclass of loss.Loss.
        :param eps: 
            A learning rate used to scale the gradient before it is used to update the model parameters.
        '''
        self.net = net
        self.loss = loss
        self.eps = eps


    def step(self, X_batch: np.ndarray, y_batch: np.ndarray,
        eval_fn=None, X_valid: np.ndarray=None, y_valid: np.ndarray=None) -> float:
        '''
        Optimizer step

        :param X_batch (numpy.ndarray): 
            The input data used to train the model.
        :param y_batch (numpy.ndarray):
            The target values corresponding to the input data.
        :param eval_fn (callable): 
            A function that calculates the evaluation metric(s) of interest (e.g. accuracy) for the model. 
            If not None, the evaluation metrics will be calculated and printed at the end of each epoch.
        :param X_valid (numpy.ndarray): 
            The validation input data used to evaluate the model during training. 
            If not None, the evaluation metrics will be calculated and printed at the end of each epoch.
        :param y_valid (numpy.ndarray): 
            The target values corresponding to the validation input data. 
            If not None, the evaluation metrics will be calculated and printed at the end of each epoch.
        '''
        # Forward pass
        y_pred = self.net(X_batch)
        loss = self.loss(y_batch, y_pred).mean()

        # Set gradient to 0
        self.net.zero_grad()

        # Backward pass
        delta = self.loss.backward(y_batch, y_pred)
        self.net.backward_update_gradient(X_batch, delta)

        # Update parameters
        self.net.update_parameters(self.eps)

        loss_valid = None
        acc_valid = None

        if X_valid is not None and y_valid is not None:
            y_pred = self.net(X_valid)
            loss_valid = self.loss(y_valid, y_pred).mean()
            
            if eval_fn is not None:
                acc_valid = eval_fn(self.net, X_valid, y_valid)

        return loss, loss_valid, acc_valid

def SGD(net: Module, loss: loss.Loss, eps: float, X: np.ndarray, y: np.ndarray, 
        epochs: int=1000, gradient_descent_mode: int=0, batch_size: int=64, 
        eval_fn=None, X_valid: np.ndarray=None, y_valid: np.ndarray=None, 
        verbose: bool=True, verbose_every: int=10):
    all_loss, all_loss_valid, all_acc_valid = [], [], []
    best_loss_valid, best_acc_valid, best_parameters = float('inf'), float('inf'), None

    N = len(X)
    if gradient_descent_mode == 2:  # STOCHASTIC
        batch_size = 1

    optim = Optim(net, loss, eps)

    def batch(X_batch, y_batch, verbose):
        nonlocal best_loss_valid, best_acc_valid, best_parameters
        print(f"Batch shape - X_batch: {X_batch.shape}, y_batch: {y_batch.shape}")
        loss, loss_valid, acc_valid = optim.step(X_batch, y_batch, eval_fn, X_valid, y_valid)

        if verbose:
            if loss_valid is None:
                print(f'Loss: {loss}')
            else:
                if acc_valid is None:
                    print(f'Train Loss: {loss}, Val Loss: {loss_valid}')
                else:
                    print(f'Train Loss: {loss}, Val Loss: {loss_valid}, Val Acc: {acc_valid}')

        all_loss.append(loss)
        if loss_valid is not None:
            all_loss_valid.append(loss_valid)
            if acc_valid is None and loss_valid < best_loss_valid:
                best_loss_valid = loss_valid
                best_parameters = optim.net.get_parameters()

        if acc_valid is not None:
            all_acc_valid.append(acc_valid)
            if acc_valid > best_acc_valid:
                best_acc_valid = acc_valid
                best_parameters = optim.net.get_parameters()

    if verbose: print('Train: -----------------------------------')
    for i in range(epochs):
        verbose_epoch = verbose and ((i + 1) % (epochs // verbose_every) == 0) if epochs > verbose_every else True
        if verbose_epoch: print(f'Epoch {i + 1}: ', end='')

        if gradient_descent_mode == 0:  # BATCH
            batch(X, y, verbose_epoch)
        else:
            idx = np.arange(N)
            np.random.shuffle(idx)
            if verbose_epoch: print()
            for bi in range(0, N, batch_size):
                bi_ = min(N, bi + batch_size)
                batch_idx = idx[bi:bi_]
                if verbose_epoch: print(f'Batch {bi_}: ', end='')
                batch(X[batch_idx], y[batch_idx], verbose_epoch)

    if verbose: print('-------------------------------------------')
    return all_loss, all_loss_valid, all_acc_valid, best_parameters
