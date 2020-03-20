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

