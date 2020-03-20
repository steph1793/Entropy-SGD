# EntropySGD

EntropySGD is a machine learning optimization method that has been pusblished in 2017 (https://arxiv.org/pdf/1611.01838.pdf). This method despite others like SGD, Adam, etc is built to ACTIVELY  search and converge towards flat region minimas, which are known to give better generalization performances than sharp region minimas.

Even though it derives from SGD theoretically, in practice, we can include improvements such as Nesterov's momentum or Adam optimization that one can decide to use or not (if one wants to stick to the original version of Entropy SGD).
In order to facilitate the usage of this method in a deep learning framework, I built a **keras** version of this optimizer.

## Dependencies

- Keras
- Tensorflow 2.0.0
- numpy
- math

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install EntropySGD.

```bash
pip install EntropySGD
```

## Usage
In this package, we have implemented 3 classes :
- **EntropySGD** : which is the original implementation of the optimizer (with SGD and optionnaly a  Nesterov's Momentum on the outer loop update (the main update))
- **EntropyAdam** : this is an adaptation of EntropySGD to Adam optimizer (on the outer loop update ).
- **History** : this is a callback used to log the training and evaluation losses. Given the particularity of this optimizer (two loops) in comparison to a more classical one as SGD , one must discard the keras logger per default and use this one. At the end of the training we can find 4 arrays:
    - loss : the training losses after each epoch
    - val_loss : the evaluation loss at the end of each epoch
    - eff_loss : or effective loss. In EntropySGD the main update is done once every L iterations (L being the number of langevin descent steps). One effective epoch corresponds then to L regular epochs. eff_loss stores then the average loss after L regular epochs.
    - eff_val_loss : effective evaluation loss.
```python
from EntropySGD import EntropySGD, EntropyAdam, History

#create your keras model
model = ...

#create your optimizer
## EntropySGD optimizer
optimizer = EntropySGD(lr=.001, sgld_step=0.1, 
                        L=20, gamma=0.03, sgld_noise=1e-4, alpha=0.75, 
                        scoping=1e-3, momentum=0., nesterov=False, decay=.0)
## or EntropyAdam optimizer
optimizer = EntropyAdam(lr=.001, sgld_step=0.1, L=20, gamma=0.03, 
                        sgld_noise=1e-4, alpha=0.75, scoping=1e-3, 
                        beta_1=0.9, beta_2=0.999, amsgrad=False, decay=0.)

# create the logger callback
history = History()

#Compile and train
model.compile(optimizer, loss = ...)
model.fit(..., callbacks=[history], verbose=0)
##very important : set the verbose to 0 to deactivate keras default logger.

print("Training loss : ", history.loss)
print("Training effective loss :", history.eff_loss)
print("Val loss", history.eff_val_loss)
print("Effective Val loss", history.eff_val_loss)
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)