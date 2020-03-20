from keras.optimizers import Optimizer
from keras import backend as K
from keras.legacy import interfaces
import tensorflow as tf
import math


import keras
import numpy as np
from tensorflow.keras.utils import Progbar
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 


class EntropySgd(Optimizer):
    """Entropy SGD optimizer. This implementation may take into account Nesterov's Momentum similar to keras implementation in SGD.
    # Arguments
        lr: float > 0. Learning rate.
        sgld_step (eta prime): float > 0. The inner sgld step size (for x' update)
        L: int > 0. Number of Langevin steps (inner loop) for the estimation of the gradient
        gamma : float > 0 . the scope allow the inner SGLD to explore further away from the parameters when estimating the negative local entropy
        scoping : float >= 0 . gamma exponential decay parameter : gamma*(1+scoping)^t
        sgld_noise : float >0. thermal noise, used in the langevin dynamics update (inner loop)
        alpha : float <1 & > 0 . Exponential averaging parameter for the estimation of mu. More details in the paper.

        momentum :  float >= 0. Parameter that accelerates SGD in the relevant direction and dampens oscillations.
        nesterov : boolean. Whether to apply Nesterov momentum.

        decay : >= 0. . Learning rate exponential decay. 0 for no decay.

    #Reference
    - [ENTROPY-SGD: BIASING GRADIENT DESCENT INTO WIDE VALLEYS](https://arxiv.org/pdf/1611.01838.pdf)
    """
    def __init__(self, lr=.001, sgld_step=0.1, L=20, gamma=0.03, sgld_noise=1e-4, alpha=0.75, scoping=1e-3, momentum=0., nesterov=False, decay=.0, **kwargs):
        super(EntropySgd, self).__init__(**kwargs)
        self.scoping = scoping
        self.momentum = momentum
        self.nesterov = nesterov
        self.L = L
        self.alpha = alpha
        self.initial_decay = decay
        with K.name_scope(self.__class__.__name__):
            self.lr = K.variable(lr, name='lr')
            self.sgld_step = K.variable(sgld_step, name='sgld_step')
            self.gamma = K.variable(gamma, name='gamma')
            self.sgld_noise = K.variable(sgld_noise, name='sgld_noise')
            
            self.state_counter = K.variable(0, dtype='int64', name='state_counter')
            self.num_steps = K.variable(-1, dtype='int32')
            self.iterator = K.variable(0, dtype='int32', name='iterator')
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.decay =  K.variable(self.initial_decay, name='decay')
        

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        
        self.updates = [];
        self.updates.append(K.update_add(self.state_counter, 1))
        self.updates.append(K.update_add(self.iterator, 1))
        self.updates.append(K.update_add(self.iterations, 1))

        lr = self.lr
        ## lr exponential decay
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        shapes = [K.int_shape(p) for p in params]
        x = [K.update(K.zeros(shape), p) for shape, p in zip(shapes, params)]
        mu = [K.update(K.zeros(shape), p) for shape, p in zip(shapes, params)]

        grads = self.get_gradients(loss, params)
        moments = [K.zeros(shape, name='moment_' + str(i))
                   for (i, shape) in enumerate(shapes)]

        for x_i, x_prime_i, mu_i, g, m in zip(x, params, mu, grads, moments):

            ## we update x_prime (if we are in LAngevin steps, we update, otherwise we switch to parameters x_i)
            dx_prime_i =  g - self.gamma*(x_i - x_prime_i)
            x_prime_update_i = K.switch(K.any(K.stack([K.equal(self.state_counter, 0), 
                                                       K.equal(self.num_steps, self.iterator)], axis=0), axis=0), 
                                        x_i,
                                        x_prime_i - self.sgld_step*dx_prime_i + K.sqrt(self.sgld_step)*self.sgld_noise*K.random_normal(K.int_shape(x_prime_i)) 
                                        )
            # Apply constraints.
            if getattr(x_prime_i, 'constraint', None) is not None:
                x_prime_update_i = x_prime_i.constraint(x_prime_update_i)
            self.updates.append(K.update(x_prime_i, x_prime_update_i))

            ## We update mu (if we are in LAngevin steps, we update otherwise we switch to parameters x_i)
            mu_update_i = K.switch(K.equal(self.state_counter, 0), 
                                   x_i,
                                   (1-self.alpha)*mu_i + self.alpha*x_prime_i)
            self.updates.append(K.update(mu_i, mu_update_i))


            ## As they described in the paper, we remove the gamma from the update because it interferes with the learning annealing
            ## After each outer loop update we apply an exponential decay on gamma
            ## The following lines concerns the outer loop updates

            ## Nesterov's momentum
            gradient = (x_i-mu_i)
            v = self.momentum * m - lr * gradient  # velocity
            self.updates.append(
              K.update(m, 
                K.switch(K.equal(self.state_counter, self.L+1), v , m ) ))
            if self.nesterov:
                new_x_i = x_i + self.momentum * v - lr * gradient
            else:
                new_x_i = x_i + v

            x_i_update = K.switch(K.equal(self.state_counter, self.L+1), new_x_i , x_i )
            self.updates.append(K.update(x_i, x_i_update))


            ## Gamma scoping
            gamma_update = K.switch(K.equal(self.state_counter, self.L+1) , self.gamma, self.gamma*(1. + self.scoping) )
            self.updates.append(K.update(self.gamma, gamma_update))


        counter = K.switch(K.equal(self.state_counter, self.L+2),  K.constant(0, dtype='int64'),self.state_counter)
        self.updates.append(K.update(self.state_counter, counter))
        return self.updates


    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'sgld_step' : float(K.get_value(self.sgld_step)),
                  'gamma' : float(K.get_value(self.gamma)),
                  'sgld_noise' : float(K.get_value(self.sgld_noise)),
                  'L' : self.L,
                  "alpha" : self.alpha,
                  'scoping' : self.scoping,
                  'momentum' : self.momentum,
                  'nesterov' : self.nesterov,
                  'decay' : self.decay}
        base_config = super(EntropySgd, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
