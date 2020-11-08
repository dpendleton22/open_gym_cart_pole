import torch
import numpy as np
import gym
import random
from matplotlib import pyplot as plt

#setup all the initial parameters
EPSILON = 1
#the rate to decay the rate of taking random actions
EP_DECAY = 0.99995
ALPHA = 0.01

#set the memory and its length
memory = []
num_memory = 1000000

episode_reward = []
epsilon_episode = []

env = gym.make('CartPole-v1')
env._max_episode_steps = 800

class model():
    def __init__(self):
        #setup the deep neural network
        self.model = torch.nn.Sequential(
            torch.nn.Linear(len(env.observation_space.sample()), 24),
            torch.nn.ReLU(),
            torch.nn.Linear(24, env.action_space.n)
        )
        self.learning_rate = 1e-3

        self.loss_func = torch.nn.MSELoss()

    def fit(self, x, y):
        y_pred = self.model(torch.Tensor(x))
        #calculate the loss from the prediction
        loss = self.loss_func(y_pred, y)
        #when updating the weights we need to zero the gradients before backpropagating
        self.model.zero_grad()
        #backpropogate the weights within the neural network
        loss.backward()
        with torch.no_grad():
            for parm in self.model.parameters():
                parm -= (self.learning_rate * parm.grad)

    def predict(self, state):
        """ Compute Q values for all actions using the DQL. """
        return self.model(torch.Tensor(state))

#replay memory
def replay_memories(model, N, gamma):
    #if the number of memories is less then the replay memory length then wait and gain exploration
    if (len(memory) < N):
        return

    sample_memories = random.sample(memory, N)

    sample_memories = np.asanyarray(sample_memories, list, order=(4,))

    # grab the state from all the memories
    #The values needed to be reshaped into a matrix
    mem_state = np.reshape(np.hstack(sample_memories[:, 0]), (N, 4))

    # grab the action from all the memories
    mem_action = sample_memories[:, 1]

    # grab the next state from the memories
    mem_next_state = np.reshape(np.hstack(sample_memories[:, 2]), (N, 4))

    # grab the reward from the memories
    mem_reward = sample_memories[:, 3]

    #grab the booleans indicating if the memory led to ending the environment
    mem_done = sample_memories[:, 4]

    #update equation with a logical done to make game ending actions 0 reward
    update_q = mem_reward + ((gamma * torch.max(m.predict(torch.Tensor(mem_next_state)), axis=1).values).detach().numpy()) * \
               np.logical_not(mem_done)

    #replace the previous outputted values from the memories with th eupdated values
    # update function
    mem_state_q = m.predict(torch.Tensor(mem_state))

    mem_action = mem_action.astype('int')
    mem_state_q = mem_state_q.detach().numpy()
    mem_state_q[np.arange(0, N), mem_action] =  update_q

    m.fit(torch.Tensor(mem_state), torch.Tensor(mem_state_q))

#initialize the model
m = model()

# set the initial training run
for i in range(0, 1500):
    #check what the number of observations are in the env
    state = env.reset()

    ep_reward = 0
    # set the looped runs within the training run
    while True:
        if i >= 1499:
            env.render()
        # check if the policy will be epsilon greedy
        #if random is greater than epsilon then take a random action
        if np.random.random() < EPSILON:
            action = np.random.randint(env.action_space.n)
        #else take the argmax from the state the agent is in
        else:
            action = int(torch.argmax(m.predict(torch.Tensor(state))))
        # take an action
        next_state, reward, done, _ = env.step(action)

        EPSILON = EPSILON * EP_DECAY
        if EPSILON <= 0.01:
            EPSILON = 0.01

        ep_reward += reward

        # add this to memory
        # a memory contains the state the agent was in, the action taken,
        # the state the agent landed in after the action and the reward from taking the action
        memory.append([state, action, next_state, reward, done])

        # if memory length if greater than designated memory length then grab the last memories that match the
        # designated mmeory size
        if (len(memory) > num_memory):
            memory = memory[-num_memory:]

        #replay memories
        replay_memories(m, 20, 0.95)

        state = next_state

        #check if the game has ended
        # if done reset environment
        if done:
            print ("Run: {} - Epsilon: {} - Reward: {}".format(i, EPSILON, ep_reward))
            episode_reward.append(ep_reward)
            epsilon_episode.append(EPSILON)
            break

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(episode_reward)
ax2.plot(epsilon_episode)
plt.show()
