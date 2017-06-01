## Overview
This is the code for the SF Python meetup group tutorial on reinforcement learning. We will build the game of [Pong](http://www.ponggame.org) using Pygame and then build a [Deep Q Network](https://www.quora.com/Artificial-Intelligence-What-is-an-intuitive-explanation-of-how-deep-Q-networks-DQN-work) using Tensorflow. Then we will train the network to play the game. The DQN is a convolutional neural network that reads in pixel data from the game and the game score. 

## Installation on python3
1. Run `python3 -m venv env`
2. Run `source env/bin/activate`
3. Run `pip install -r requirements.txt`

## Installation on Anaconda
1. Install Continuum miniconda (https://conda.io/miniconda.html)
2. Run `conda env create`
3. Run `source activate pong`

This should install all necessary dependencies in a painless way.

*NOTE for windows users*: you have to edit the `environment.yml` and change python version from 3.6 to 3.5. This is because Tensorflow does not offer a windows python 3.6 version.

For manual install, here are the required dependencies and links to install them:

- numpy>=1.7
- pygame>=1.9.3 (https://www.pygame.org/download.shtml)
- tensorflow>=1.0.0 (https://www.tensorflow.org/install/)

## Usage 
Once you've completed the exercises, you can run it like in terminal:
```
python RL.py
```
The longer you let it run, the better it will get.

## Solutions
Solution code is provided in the solutions folder.

## Credits

Code originally developed by [malreddysid](https://github.com/malreddysid), updated by [llSourcell](https://github.com/llSourcell). We've adapted it to TF 1.0, Anaconda python, removed OpenCV dependency and adapted it to be used for an exercise.

## References
- http://karpathy.github.io/2016/05/31/rl/
- http://rll.berkeley.edu/deeprlcourse/
- http://www.wildml.com/2016/10/learning-reinforcement-learning/
- https://www.nervanasys.com/demystifying-deep-reinforcement-learning/
