# coding: utf-8

import numpy as np
import pickle
import gym

# hyperparameters
H = 10 # number of hidden layer neurons
learning_rate = 1e-4 # 0.0001 = 1 / 10000
resume = False # resume from previous checkpoint?
render = False

# model initialization
if resume:
    model = pickle.load(open('model.p', 'rb'))
else:
    model = {}
    model['W1'] = np.random.randn(H) # "Xavier" initialization
    model['W2'] = np.random.randn(H) / np.sqrt(H)
    pickle.dump(model, open('model.p', 'wb'))


def sigmoid(x, deriv = False):
    f = 1.0 / (1.0 + np.exp(-x))
    if (deriv == False):
        return f
    else:
        return f * (1 - x)


def prepro(I):
    """ prepro 210 x 160 x 3 uint8 frame into one number """
    I = I[35:190, 20:140, 0] # crop 210x160x3 -> 155x120x1
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    I = np.transpose(np.nonzero(I))
    
    if (I.size != 0):
        x = I[1, 0]
        return x / 155.
    else:
        return 0.5


def policy_forward(x):
    l1 = x * model['W1'] # число * вектор -> вектор
    l1[l1 <= 0] = 0 # ReLU nonlinearity
    l2 = np.dot(model['W2'], l1) # вектор * вектор -> число
    l2 = sigmoid(l2)
    return l1, l2 # return probability of taking action 2


def policy_backward(xs, loss, l1s):
    delta_cache = loss * model['W2']
    for x in xs:
        for i in range(H):
            if (model['W1'][i] > 0):
                model['W1'][i] += delta_cache * x * learning_rate
    for l1 in l1s:
        l2 = np.dot(model['W2'], l1)
        for i in range(H):
            model['W2'][i] += loss * learning_rate * l1[i] * sigmoid(l2, deriv = False)

        
env = gym.make("Pong-v0")
observation = env.reset()
coords, l1s = [], []
delta_cache = [] # delta memory

while True:
    if render:
        env.render()

    # preprocess the observation, set input to network to be ball coordinates
    coord = prepro(observation)

    # forward the policy network and sample an action from the returned probability
    l1, l2 = policy_forward(coord)
        
    if (l2 > 0.5):
        action = 2
    elif (l2 < 0.5):
        action = 3
    else:
        action = 1

    # record various intermediates (needed later for backprop)
    coords.append(coord) # observation      
    l1s.append(l1)

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    
    if (reward != 0): # an episode finished
        
        # robocraft.ru/blog/algorithm/560.html
        policy_backward(coords, reward, l1s)
        pickle.dump(model, open('model.p', 'wb'))

        coords, l1s = [], [] # reset array memory
        observation = env.reset() # reset env
    
        # Pong has either +1 or -1 reward exactly when game ends
        print(('Game finished, reward: %f' %reward) + ('' if reward == -1 else ' !!!!!!!!'))
