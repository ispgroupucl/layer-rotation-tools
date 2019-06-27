'''
Code for applying Layca on SGD, Adam, RMSprop and Adagrad.
Source: code is based on keras' implementation of the original optimization methods.
'''

from keras.optimizers import Optimizer
import keras.backend as K
from keras.legacy import interfaces

import numpy as np

def norm(w):
    '''
    computes frobenius norm of a tensor w
    '''
    return K.sqrt(K.sum(w*w))

def layca(p, step, lr):
    '''
    Core operations of layca.
    Takes the current parameters and the step computed by an optimizer, and 
         - projects and normalizes the step such that the rotation operated on the layer's weights is controlled
         - after the step has been taken, recovers initial norms of the parameters
         
    !!! 
    only kernels are optimized. Biases and batchnorm pararmeters are left unchanged. This did not affect performance in our experiments. 
    One can decide to train them anyway (without layca operations) by changing last line to:
    return p  - lr * step
    !!!
    '''
    if 'kernel' in p.name: # only kernels are optimized when using Layca (and not biases and batchnorm parameters)
        # projecting step on tangent space of sphere -> orthogonal to the parameters p
        initial_norm = norm(p)
        step = step - (K.sum(step * p))* p / initial_norm**2

        # normalizing step size (with special attention to numerical problems)
        step = K.switch(norm(step)<= K.epsilon(), lambda: K.zeros_like(step), lambda: step/ (norm(step)) * initial_norm)
        
        # applying step
        new_p =  p - lr * step

        # recovering norm of the parameter from before the update
        new_p = new_p / norm(new_p) * initial_norm
        return new_p
    else:
        return p # - lr * step  # uncomment to train biases and batchnorm parameters (without layca)
            
class SGD(Optimizer):
    """Stochastic gradient descent optimizer.
    Includes support for momentum,
    learning rate decay, and Nesterov momentum.
    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter that accelerates SGD
            in the relevant direction and dampens oscillations.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
        layca: boolean, wether to apply layca or not
    """

    def __init__(self, lr=0.01, momentum=0., decay=0.,
                 nesterov=False, layca = False,**kwargs):
        super().__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay
        self.nesterov = nesterov
        
        self.layca = layca

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):        
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):  
            v = self.momentum * m - lr * g  # velocity
            self.updates.append(K.update(m, v))
             
            if self.nesterov:
                step =  self.momentum * v - lr * g
            else:
                step =  v
            
            if self.layca:
                new_p = layca(p, -step, lr)
            else:
                new_p =  p + step

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov,
                  'layca':self.layca}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

class RMSprop(Optimizer):
    """RMSProp optimizer.
    It is recommended to leave the parameters of this optimizer
    at their default values
    (except the learning rate, which can be freely tuned).
    This optimizer is usually a good choice for recurrent
    neural networks.
    # Arguments
        lr: float >= 0. Learning rate.
        rho: float >= 0.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        layca: boolean, wether to apply layca or not
    # References
        - [rmsprop: Divide the gradient by a running average of its recent magnitude](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
    """

    def __init__(self, lr=0.001, rho=0.9, epsilon=None, decay=0., layca = False,
                 **kwargs):
        super(RMSprop, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.lr = K.variable(lr, name='lr')
            self.rho = K.variable(rho, name='rho')
            self.decay = K.variable(decay, name='decay')
            self.iterations = K.variable(0, dtype='int64', name='iterations')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        
        self.layca = layca

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):        
        grads = self.get_gradients(loss, params)
        accumulators = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = accumulators
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))

        for p, g, a in zip(params, grads, accumulators):
            # update accumulator
            new_a = self.rho * a + (1. - self.rho) * K.square(g)
            self.updates.append(K.update(a, new_a))
            #new_p = p - lr * g / (K.sqrt(new_a) + self.epsilon)
            step = lr * g / (K.sqrt(new_a) + self.epsilon)
            
            if self.layca:
                new_p = layca(p, step, lr)
            else:
                new_p =  p - step
            
            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'rho': float(K.get_value(self.rho)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'layca':self.layca}
        base_config = super(RMSprop, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class Adam(Optimizer):
    """Adam optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        layca: boolean, wether to apply layca or not
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".
    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
        - [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, layca = False,
                 epsilon=None, decay=0., amsgrad=False, **kwargs):
        super(Adam, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad
        
        self.layca = layca
    
    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros((1,)) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                step = lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                step = lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            
            if self.layca:
                new_p = layca(p, step, lr)
            else:
                new_p =  p - step

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'amsgrad': self.amsgrad,
                  'layca':self.layca}
        base_config = super(Adam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Adagrad(Optimizer):
    """Adagrad optimizer.
    It is recommended to leave the parameters of this optimizer
    at their default values.
    # Arguments
        lr: float >= 0. Learning rate.
        epsilon: float >= 0. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        layca: boolean, wether to apply layca or not
    # References
        - [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
    """

    def __init__(self, lr=0.01, epsilon=None, decay=0., layca = False,**kwargs):
        super(Adagrad, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.lr = K.variable(lr, name='lr')
            self.decay = K.variable(decay, name='decay')
            self.iterations = K.variable(0, dtype='int64', name='iterations')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        
        self.layca = layca

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        
        grads = self.get_gradients(loss, params)
        shapes = [K.int_shape(p) for p in params]
        accumulators = [K.zeros(shape) for shape in shapes]
        self.weights = accumulators
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))

        for p, g, a in zip(params, grads, accumulators):
            new_a = a + K.square(g)  # update accumulator
            self.updates.append(K.update(a, new_a))
            step = lr * g / (K.sqrt(new_a) + self.epsilon)
            
            if self.layca:
                new_p = layca(p, step, lr)
            else:
                new_p =  p - step
            
            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'layca':self.layca}
        base_config = super(Adagrad, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))