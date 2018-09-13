# Spaceship-RL
Implementation of various Reinforcement Learning models in the context of a spaceship navigation task [Python / PyTorch]

Task of agent is to navigate a spaceship to a beacon in a 2D solar system, while avoiding collision with the star and its orbiting planets.

Currently implemented RL models:

- Random walker
- Monte-Carlo sampler
- TD-lambda learner (of Q(s,a), on-policy)
- FFNN (feed-forward neural network learning Q(s,a) value function by error back-propagation)
- RNN (recurrent neural network performing generative model-based planning, TODO)

Implemented analysis methods: 

- Connectivity maps
- Feature selectivity (to position, distance, angle, time, etc)
- Prediction error of environmental variables (position and size of planets, star and ship, etc)
