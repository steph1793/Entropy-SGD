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


class ESGD(Optimizer):
    """Entropy SGD optimizer
    # Arguments
        lr: float >= 0. Learning rate.
        sgld_step (eta prime): float > 0. The inner sgld step size
        L: int > 0. Number of Langevin steps.
        gamma : float >0 . the scope allow the inner SGLD to explore further away from the parameters
        epsilon : float >0. thermal noise
    #Reference
    - [ENTROPY-SGD: BIASING GRADIENT DESCENT INTO WIDE VALLEYS](https://arxiv.org/pdf/1611.01838.pdf)
    """
    def __init__(self, lr=1., sgld_step=0.1, L=20, gamma=0.03, epsilon=1e-4, alpha=0.75, scoping=1e-3, momentum=0., nesterov=False, **kwargs):
        super(ESGD, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.scoping = scoping
            self.momentum = momentum
            self.nesterov = nesterov
            self.lr = K.variable(lr, name='lr')
            self.sgld_step = K.variable(sgld_step, name='sgld_step')
            self.gamma = K.variable(gamma, name='sgld_step')
            self.epsilon = K.variable(epsilon, name='sgld_step')
            self.L = L
            self.state_counter = K.variable(0, dtype='int64', name='state_counter')
            self.alpha = alpha

            self.num_steps = K.variable(-1, dtype='int32')
            self.iterator = K.variable(0, dtype='int32', name='state_counter')
        

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        
        self.updates.append(K.update_add(self.state_counter, 1))
        self.updates.append(K.update_add(self.iterator, 1))
        lr = self.lr
        shapes = [K.int_shape(p) for p in params]
        x = [K.update(K.zeros(shape), p) for shape, p in zip(shapes, params)]
        mu = [K.update(K.zeros(shape), p) for shape, p in zip(shapes, params)]

        grads = self.get_gradients(loss, params)

        moments = [K.zeros(shape, name='moment_' + str(i))
                   for (i, shape) in enumerate(shapes)]

        for x_i, x_prime_i, mu_i, g, m in zip(x, params, mu, grads, moments):

            ## we update x_prime (if we are in LAngevin steps, we update otherwise we switch to parameters x_i)
            dx_prime_i =  g - self.gamma*(x_i - x_prime_i)
            x_prime_update_i = K.switch(K.any(K.stack([K.equal(self.state_counter, 0), 
                                                       K.equal(self.num_steps, self.iterator)], axis=0), axis=0), 
                                        x_i,
                                        x_prime_i - self.sgld_step*dx_prime_i + K.sqrt(self.sgld_step)*self.epsilon*K.random_normal(K.int_shape(x_prime_i)) 
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

            ## We update x every L steps (Note that at step L+1 or when step < L, the update term is 0. This is coherent with the paper)
            ## As they described in the paper, we remove the gamma from the update because it interferes with the learning annealing
            ## After each update we rescale gamme with a factor of 1.001


            ## Momentum and Nesterov
            v = self.momentum * m - lr * (x_i-mu_i)  # velocity
            self.updates.append(K.update(m, v))
            if self.nesterov:
                new_x_i = x_i + self.momentum * v - lr * (x_i-mu_i)
            else:
                new_x_i = x_i + v

            x_i_update = K.switch(K.equal(self.state_counter, self.L+1), new_x_i , x_i )
            self.updates.append(K.update(x_i, x_i_update))


            ## Gamma scoping
            gamma_update = K.switch(self.state_counter<self.L , self.gamma, self.gamma*(1. + self.scoping) )
            self.updates.append(K.update(self.gamma, gamma_update))


        counter = K.switch(K.equal(self.state_counter, self.L+2),  K.constant(0, dtype='int64'),self.state_counter)
        self.updates.append(K.update(self.state_counter, counter))
        return self.updates


    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'sgld_step' : float(K.get_value(self.sgld_step)),
                  'gamma' : float(K.get_value(self.gamma)),
                  'epsilon' : float(K.get_value(self.epsilon)),
                  'L' : int(K.get_value(self.L))}
        base_config = super(SGLD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))







class History(keras.callbacks.Callback):

  def __init__(self):
    super(History, self).__init__()
    self.loss = []
    self.val_loss = []
    self.eff_loss = []
    self.eff_val_loss = []
    self.i = 0


  def on_train_begin(self, logs=None):
    self.epochs = self.params['epochs']
    K.set_value(self.model.optimizer.num_steps, 
                math.ceil(self.params["samples"]/self.params["batch_size"])*self.params["epochs"])
  
  def on_epoch_begin(self, epoch, logs={}):
    self.loss_buff = []
    self.val_loss_buff = []
    print('Epoch : %d/%d, Effective Epoch : %d/%d' % (epoch + 1, self.epochs, (epoch+1)//self.model.optimizer.L+1, self.epochs//self.model.optimizer.L))
    self.target = self.params['samples']
    self.progbar = Progbar(target=self.target,
                              verbose=1,
                              stateful_metrics=['loss', 'val_loss'])
    self.seen = 0

  def on_train_batch_begin(self, batch, logs=None):
      if self.seen < self.target:
          self.log_values = []


  def on_train_batch_end(self, batch, logs={}):
    self.i = self.i+1
    batch_size = logs.get('size', 0)
    self.seen += batch_size
    
    if K.eval(self.model.optimizer.state_counter) == 0:
      self.loss_buff.append(logs.get('loss'))
      self.log_values.append(('loss', np.mean(self.loss_buff)))

    # Skip progbar update for the last batch;
    # will be handled by on_epoch_end.
    if self.seen < self.target:
        self.progbar.update(self.seen, self.log_values)
    else:
        self.progbar.update(self.target-1, self.log_values)



  def on_test_batch_end(self, batch, logs={}):
    self.val_loss_buff.append(logs.get('loss'))
    


  def on_epoch_end(self, epoch, logs):
    self.loss.append(np.mean(self.loss_buff))
    self.val_loss.append(np.mean(self.val_loss_buff))

    if (epoch+1)%self.model.optimizer.L == 0:
      self.eff_loss.append(np.mean(self.loss[-self.model.optimizer.L:]))
      self.eff_val_loss.append(np.mean(self.val_loss[-self.model.optimizer.L:]))
      
    self.log_values.append(('val_loss', np.mean(self.val_loss_buff)))
    self.progbar.update(self.target, self.log_values)

