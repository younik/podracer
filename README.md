## Podracer - Distributed DRL

Implementation of Sebulba framework from [Podracer architectures for scalable Reinforcement Learning](https://arxiv.org/pdf/2104.06272.pdf) by Hessel et al.


Currently implementing PPO with code adapted from [vwxyzjn/cleanba](https://github.com/vwxyzjn/cleanba)


### 🚧 Modular blueprint:

| Module    | Description                                         |
|-----------|-----------------------------------------------------|
| policy    | Implements the neural network in Flax               |
| agent     | Uses the policy to collect rollouts                 |
| learner   | Implements the learning algorithm (e.g. PPO)        |
| framework | Implements the framework (e.g. Sebulba or Anakin)   |
| optimizer | Handles the Optax optimizer creation                |
| env       | Handles environment creation                        |
| args      | Arguments files to easily switch across experiments |
| launch    | Setup the arguments and WandB run                   |