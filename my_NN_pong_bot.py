# coding: utf-8

""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
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
  
delta_cache = { 'delta_layer1': np.zeros_like(model['W2'])} # delta memory
grad_cache = { 'grad_layer1': np.zeros_like(model['W2'])} # grad memory


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


def policy_forward(x, model):
    l1 = x * model['W1'] # число * вектор -> вектор
    l1[l1 <= 0] = 0 # ReLU nonlinearity
    l2 = np.dot(model['W2'], l1) # вектор * вектор -> число
    l2 = sigmoid(l2)
    return l1, l2 # return probability of taking action 2, and hidden state


def loss_backward(loss, model_W2):
    delta_layer1 = loss * model_W2 # число * вектор -> вектор
    return delta_layer1


env = gym.make("Pong-v0")
observation = env.reset()
coords, l1s, l2s = [], [], []

while True:
    if render:
        env.render()

    # preprocess the observation, set input to network to be ball coordinates
    coord = prepro(observation)

    # forward the policy network and sample an action from the returned probability
    l1, l2 = policy_forward(coord, model)
        
    if (l2 > 0.5):
        action = 2
    elif (l2 < 0.5):
        action = 3
    else:
        action = 1

    # record various intermediates (needed later for backprop)
    coords.append(coord) # observation
    l1s.append(l1) # hidden state        
    l2s.append(l2)

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    
    if (reward != 0): # an episode finished

        loss = reward

        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epcoord = np.vstack(coords)
        epl1 = np.vstack(l1s)
        epl2 = np.vstack(l2s)

        coords, l1s, l2s = [], [], [] # reset array memory
        
        # robocraft.ru/blog/algorithm/560.html
        delta_cache['delta_layer1'] = loss_backward(loss, model['W2'])
        grad_cache[]
        
        # edit model
        model['W1'] += learning_rate * model[k]

        # perform rmsprop parameter update every episode
        for k, v in model.iteritems():
            g = grad_buffer[k] # gradient
            rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
            model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
            grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer
            pickle.dump(model, open('model.p', 'wb'))

        observation = env.reset() # reset env
    
    # Pong has either +1 or -1 reward exactly when game ends.
    print(('game finished, re+ward: %f' %reward) + ('' if reward == -1 else ' !!!!!!!!'))

