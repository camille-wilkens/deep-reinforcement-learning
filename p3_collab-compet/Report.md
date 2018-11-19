[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"
[image3]: https://github.com/ShubraChowdhury/DeepReinforcementLearning/blob/master/p3_collab-compet/DDGP_ALGO.PNG "Algo"
[image4]: https://github.com/ShubraChowdhury/DeepReinforcementLearning/blob/master/p3_collab-compet/PLOT.PNG "Plot"


# REPORT For Project 3: Collaboration and Competition

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



### Implementation Details

There are 3 main files ddpg_agent.py and model.py, and  Tennis.ipynb. 

1. model.py: Architecture and logic for the neural networks implementing the actor and critic for the chosen DDPG algorithm.
    Actor model has 2 fully connected layer (of 400 and 300 units) and Critic has 2 fully connected layer (of 400, 300 units).Input and output layers sizes are determined by the state and action space.

Actor Model/Network architecture | Value
--- | ---
fc1_units | 400  
fc2_units | 300 

Critic Model/Network architecture | Value
--- | ---
fc1_units | 400  
fc2_units | 300 

    
2. ddpg_agent.py: This program implements Agent class and OUNoise, Agent includes step_and_buff() which saves experience in replay memory and use random sample from buffer to learn, act() which returns actions for given state as per current policy, learn() which Update policy and value parameters using given batch of experience tuples  which is used  to train the agent, and uses 'model.py' to generate the local and target networks for the actor and critic.

3. Tennis.ipynb: Contains instructions for how to use the Unity ML-Agents environment,the environments contain brains which are responsible for deciding the actions of their associated agents. Here we check for the first brain available, and set it as the default brain we will be controlling from Python.In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. The main training loop creates an agent and trains it using the DDPG (details below) until satisfactory results. 

	Saving Model: Notebook saves model for both actor and critic for both agents
	1. Saved after every 100 episode actor_checkpoint_0.pth, actor_checkpoint_1.pth and critic_checkpoint_0.pth , critic_checkpoint_1.pth
	


### Learning Algorithm

The agent is trained using the DDPG algorithm which is an off-policy algorithm. It is not possible to straightforwardly apply Q-learning to continuous action spaces, because in continuous spaces finding the greedy policy requires an optimization of at at every timestep; this optimization is too slow to be practical with large, unconstrained function approximators and nontrivial action spaces. Instead DDPG is an actor-critic approach based on the DPG algorithm (Silveret al., 2014).
The DPG algorithm maintains a parameterized actor function {mu(s|theta to power mu)} which specifies the current policy by deterministically mapping states to a specific action. The critic Q(s; a) is learned using the Bellman equation as in Q-learning. The actor is updated by following the applying the chain rule to the expected return from the start distribution J with respect to the actor parameters.

As with Q learning, introducing non-linear function approximators means that convergence is no longer guaranteed. However, such approximators appear essential in order to learn and generalize on large state spaces. NFQCA (Hafner & Riedmiller, 2011), which uses the same update rules as DPG but with neural network function approximators, uses batch learning for stability, which is intractable for large networks. A minibatch version of NFQCA which does not reset the policy at each update, as would be required to scale to large networks, is equivalent to the original DPG. DDGP is modifications to DPG, inspired by the success of DQN, which allow it to use neural network function approximators to learn in large state and action spaces online.

One challenge when using neural networks for reinforcement learning is that most optimization algorithms assume that the samples are independently and identically distributed. When the samples are generated from exploring sequentially in an environment this assumption no longer holds. Additionally, to make efficient use of hardware optimizations, it is essential to learn in minibatches, rather than online.

As in DQN, DDPG uses replay buffer to address these issues. The replay buffer is a finite sized cache R. Transitions were sampled from the environment according to the exploration policy and the tuple (st; at; rt; st+1) was stored in the replay buffer. When the replay buffer was full the oldest samples were discarded. At each timestep the actor and critic are updated by sampling a minibatch uniformly
from the buffer. Because DDPG is an off-policy algorithm, the replay buffer can be large, allowing the algorithm to benefit from learning across a set of uncorrelated transitions.

A major challenge of learning in continuous action spaces is exploration. An advantage of off policies  algorithms such as DDPG is that it can treat the problem of exploration independently from the learning algorithm.

###  Sudo Code Explanation 
![Algo][image3]


###   Training using Deep Deterministic Policy Gradient (DDPG)          
		def ddpg(n_episodes=5000, max_t=1000,print_every=100,window_size=100):
		    scores_deque = deque(maxlen=window_size) # last 100 scores
		    scores_list = []
		    average_score_list = []
		    excess_episode =500
		    excess_average_score = 0.5
		    env_solv = False
		    episode_required = 0 # Go 100 more episode beyond episode when environment is solved ==n_episodes
		    maximum_average_score = 0 # maximum average score including episode_required
		    maximum_episode = 0 # episode at which i get maximum_average_score

		    start_time = time.time()
		    for i_episode in range(1, n_episodes+1):
			scores = np.zeros(num_agents)
			env_info = env.reset(train_mode=True)[brain_name]
			states = env_info.vector_observations                  # get the current state (for each agent)
			agent.reset()

			for ts in range(max_t):
			    actions = agent.act(states)
			    env_info = env.step(actions)[brain_name]
			    next_states = env_info.vector_observations         # get next state (for each agent)
			    rewards = env_info.rewards                         # get reward (for each agent)
			    dones = env_info.local_done                        # see if episode finished


			    agent.step_and_buff(states, actions, rewards, next_states, dones,ts)
			    states  = next_states
			    scores += rewards                                  # update the score (for each agent)
			    if np.any(dones):                                  # exit loop if episode finished
				break

			score = np.max(scores)        
			scores_list.append(score)
			scores_deque.append(score)

			average_score = np.average(scores_deque)
			average_score_list.append(average_score)   

			print("\rEpisode: {:4d}   Episode Score: {:.2f}   Average Score: {:.4f}".format(i_episode,score,average_score), end="")
			if i_episode >= 100:
			    if not env_solv:
				if average_score >= 0.5:
				    end_time = time.time()
				    print("........Environment solved", "in  time {:.2f}".format( end_time-start_time))
				    episode_required = i_episode + excess_episode
				    env_solv = True
			    elif maximum_average_score < average_score:
				maximum_average_score = average_score
				maximum_episode = i_episode


			if i_episode % print_every == 0:
			    print("\rEpisode: {:4d}   Episode Score: {:.2f}   Average Score: {:.4f}".format(i_episode,score,average_score))
			    for idx, agent_name in enumerate(agent_object):
				torch.save(agent_name.actor_local.state_dict(), "actor_checkpoint_" + str(idx) + ".pth")
				torch.save(agent_name.critic_local.state_dict(), "critic_checkpoint_" + str(idx) + ".pth")            

			if i_episode >= episode_required and average_score + excess_average_score < maximum_average_score:
				break
		    print()            
		    print("\n\rMaximum Average Score (over 100 episodes): {:.4f}  at Episode: {:4d}".format(maximum_average_score,maximum_episode))

		    return scores_list,average_score_list

    
### Hyperparameters:

Parameters | Value
--- | ---
Replay buffer size | int(1e9)
Minibatch size | 1024
Discount factor | 0.99  
Tau (soft update) | 1e-3
Learning rate actor | 1e-4
Learning rate critic | 1e-3
L2 weight decay | 0.0001
Noise Sigma | 0.2
Theta | 0.15
Mu | 0

Training Parameters | Value
--- | ---
Number of episodes | 5000
Max_t | 1000
Print every |100
Deque Window |100 




### Training Output With Average Scores


Episodes | Episode Score |Average Score
--- | --- | --- 
--- | --- | --- 
  100   | 0.00   | 0.0056
  200   | 0.00   | 0.0010
  300   | 0.00   | 0.0000
  400   | 0.00   | 0.0000
  500   | 0.09   | 0.0225
  600   | 0.20   | 0.0460
  700   | 0.10   | 0.0867
  800   | 0.10   | 0.1425
  900   | 0.20   | 0.1836
 1000   | 0.10   | 0.2382
 1077   | 2.60   | 0.5066........Environment solved in  time 1912.06
 1100   | 2.60   | 0.9401
 1200   | 2.60   | 1.6532
 1300   | 2.60   | 1.9366
 1400   | 0.39   | 1.7060
 1500   | 1.49   | 1.9726
 1600   | 0.10   | 1.9760
 1636   | 0.80   | 1.6327


Environment solved in 1077 episodes!	Average Score: 0.5066, total training time: 1912.06 seconds

Maximum Average Score (over 100 episodes): 2.1498  at Episode: 1542

### Training Reward/Score Plot
![Plot][image4]

###  Ideas for future work

First I will have to do a major improvement  with hyperparameters , check  what can be the  replay buffer size for consistent maximum scores. 

Second I will work on improving the model so that when its run on Windows and Linux it should produce the same results (at this point my Linux model fails to perform well when run on windows  and same is true  with windows model).

Third implementing Prioritized Experience Replay(https://arxiv.org/abs/1511.05952?context=cs), (https://github.com/qfettes/DeepRL-Tutorials)


### References:
1. [DDPG Paper](https://arxiv.org/pdf/1509.02971.pdf)

2. [DDPG-pendulum implementation](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum)

3. Udacity code guidance for agent and model (ddpg_agent.py and model.py)

4. Reinforcement Learning Book by Richard S. Sutton  and Andrew G. Barto

5. [Silver Lever Nicolas Heess, Thomas Degris, Daan Wierstra, Martin Riedmiller ](http://proceedings.mlr.press/v32/silver14.pdf)

6.[TRPO & PPO](https://medium.com/@sanketgujar95/trust-region-policy-optimization-trpo-and-proximal-policy-optimization-ppo-e6e7075f39ed)

7. [A3C](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2)
