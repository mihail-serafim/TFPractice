import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# This hidden markov algorithm models a simple weather system
# and predicts the temperature given the following rules:
# 1. cold days are 0 and warm days are 1
# 2. the first day has an 80% chance of being cold
# 3. a cold day has a 30% chance of having a hot day next
# 4. a hot day has a 20% chance of having a cold day next
# 5. temperature is normally distributed w/:
#                           (mean, standard dev.) = (0,5) for a cold day, and
#                           (mean, standard dev.) = (15,10) for a hot day    

tfd = tfp.distributions

# set initial, transition, and observation parameters
initial_distribution = tfd.Categorical(probs=[0.2, 0.8])  # Satisfies #2 above
transition_distribution = tfd.Categorical(probs=[[0.5, 0.5], [0.2, 0.8]])  # satisfies #3 and 4 above
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])  # satisfies #5 above

days = int(input("How many days in the future would you like to predict the average temperature for?"))
# cretaing the HMM, gets distribution for next n days
model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=days)

mean = model.mean()
with tf.compat.v1.Session() as sess:  
  print("Average temperatures:", mean.numpy())
