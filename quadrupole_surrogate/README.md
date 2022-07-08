# Quadrupole Surrogate Model

This is a simple example for training an artificial neural network (ANN) for regression in a supervised setup.

Specifically, we want to train a surrogate model for a quadrupole magnet. You might want to do this for more complex systems that are either expensive to simulate or must be measured on real hardware, when you need a faster and more accessible proxy.

Our quadrupole magnet simulation takes as input a beam described by 11 values, a quadrupole length and a quadrupole strength. The simulation then outputs 11 parameters of a beam. Our ANN is designed to replicate this interface and should ideally compute the same function. To this end, in `surrogate_tf.ipynb` we demonstrate a simple training setup for this problem using *Keras* and *TensorFlow* as well as how to use *Weights & Biases (W&B)* to help you monitor the training. In `sweep.ipynb` we extend the same setup to perform a hyperparameter search using *W&B*'s Sweep feature. In `surrogate_torch.ipynb` we do the same training in *PyTorch* and introduce some more advanced concepts to the implementation including the use of *PyTorch-Lightning* to make our lives a little easier.
