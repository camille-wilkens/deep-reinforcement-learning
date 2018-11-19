[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"
[image3]: https://github.com/camille-wilkens/deep-reinforcement-learning/tree/master/p3_collab-compet/ddpg.PNG "DDPG"
[image4]: https://github.com/camille-wilkens/deep-reinforcement-learning/tree/master/p3_collab-compet/graph.PNG "Plot"


#Project 3: Collaboration and Competition

### Introduction

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment..

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

#### Solve Status
The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.


### Learning Algorithm

DDPG Algorithm The Deep Deterministic Policy Gradient (DDPG) is an actor-critic algorithm.   The input of the actor network is the current state, and the output is a single real value representing an action chosen from a continuous action space. The critic’s value model output is the estimated Q-value of the current state and of the action given by the actor. The deterministic policy gradient theorem provides the update rule for the weights of the actor network [1] & [2].  

![DDPG][image3]

### DDPG Model 
 
The Actor’s model I used consisted of two hidden linear layers with 400 and 300 units.  The output layer used was tanh. The Critic’s model consisted of three hidden layers - 400, 300 & 100 units.  Each layer is fully connected with RELU activations with an output layer consisting of 1.   Batch normalization was applied on both the Critic’s & Actor’s model. 

The hyperparameters that were used in the model consisted of: 
BUFFER_SIZE = int(1e9)    # replay buffer size 
BATCH_SIZE = 1024         # minibatch size 
GAMMA = 0.99              # discount factor 
SIGMA=0.05 TAU = 1e-3     # for soft update of target parameters 
LR_ACTOR = 1e-3           # learning rate of the actor  
LR_CRITIC = 1e-3          # learning rate of the critic 
WEIGHT_DECAY = 0          # L2 weight decay 
n_episodes=300       # of episodes 
 

### Plot of Rewards 
The DDPG algorithm achieved an average score of .5077 in 878 episodes as seen below: 
![Plot][image4]

### Improvements 
 
The areas of improvement would be to continue to tune the hyper parameters and also to implement a PPO model. 
 
### References 
[1] https://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html
[2] https://arxiv.org/abs/1509.02971 
