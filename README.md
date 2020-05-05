# Correlated Q Learning

Correlated Q Learning

## Getting Started

This the repository for application of correlated Q Learning.

### Prerequisites

What things you need to install the software

```
Numpy, matplotlib, cvxpy
```


## Running the process
soccer_game.py contains the environment implementation.

utils.py contains the plot functions and default setting of plt.

There are in total 5 different models can run, they all extends from base_learner.py, where default parameters are set.

sarsa.py contains the model for Sarsa, to run the model, just simply run __main__.

q_learning.py contains the model for off policy q learning, __main__ is where the model is run.

friend_q_learning.py is the model for friend Q learning, __main__ is where the model is run.

foe_q_learning.py is the model for Foe Q learning, __main__ is where the model is run.

ce_q_learning.py is the model for correlated Q learning, __main__ is where the model is run.

## Authors

* **Sizhang Zhao** - *Initial work* 


