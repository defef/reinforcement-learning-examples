#https://github.com/defef/reinforcement-learning-examples
import numpy as np
import random
import gym
from keras.models import Sequential
from keras.layers import Dense, Activation


REWARD_DISCOUNT_FACTOR = 0.99
MAX_GAME_STEPS = 500

def create_action_model():
    model = Sequential()
    model.add(Dense(8, input_shape=(4,),  activation='sigmoid'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    print(model.summary())
    return model


def create_value_model():
    model = Sequential()
    model.add(Dense(10, input_shape=(4,),  activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())
    return model


def update_model(model, X, y):
    model.fit(X, y,
            batch_size=5,
            epochs=3,
            verbose=0,
            shuffle=True,
            )    


def choose_action(action_model, state, explore):
    probability = action_model.predict(state.reshape(1, state.size), verbose=0)[0]        
    if explore: 
        #explore/exploit - choose action randomly in proportion to it's probability
        rand = random.random()
        #[1,0] - go left (action 0 in cartpole), [0.1] - go right (action 1 in cartpole)
        action = [1.,0.] if rand<probability[0] else [0.,1.] 
    else:
        #exploit only
        #[1,0] - go left (action 0 in cartpole), [0.1] - go right (action 1 in cartpole)
        action = [1.,0.] if probability[0]>probability[1] else [0.,1.]   
    return action, probability


def get_discounted_total_rewards(rewards):
    total_discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(xrange(0, rewards.size)):
        running_add = running_add * REWARD_DISCOUNT_FACTOR + rewards[t]
        total_discounted_rewards[t] = running_add
    return total_discounted_rewards


def run_episode(env, action_model, value_model, explore):
    states = []
    actions = []
    rewards = []
    probabilities = []
    discounted_total_rewards = []
 
    state = env.reset()
    for _ in range(MAX_GAME_STEPS):
        if not explore:
            env.render()
        #choose action
        action, probability = choose_action(action_model, state, explore)
        states.append(state)
        actions.append(action)
        probabilities.append(probability)
        
        #take action if [0,1] then go left (cartpole action 0), if [1,0] then go right (cartpole action 1)
        state, reward, done, info = env.step(np.argmax(action))
        
        rewards.append(reward)
        if done:
            break
    
    if explore:
        #reshape and convert data structures
        states = np.reshape(states, (len(states), len(states[0])))
        actions = np.reshape(actions, (len(actions), len(actions[0])))
        rewards = np.array(rewards)
        probabilities = np.array(probabilities)

        #get incurred discounted rewards         
        discounted_total_rewards = get_discounted_total_rewards(rewards)
        #get predicted discounted rewards         
        predicted_discounted_total_rewards = value_model.predict(states, verbose=0).reshape(len(rewards))
        
        #update value model
        update_model(value_model, states, discounted_total_rewards)
        
        #negative advantage means that the discounted reward for the action
        #was lower than expected by the value-model 
        advantages = discounted_total_rewards - predicted_discounted_total_rewards
        
        #determine in which direction to change the gradients.
        #first, identify which actions should've been picked according to the advantages.
        #if advantage is negative then we should increase the probability of the action that we did not take.
        ys = actions
        advantages = np.repeat(advantages.reshape(rewards.size,1), 2, axis=1)
        ys[advantages[:,0]<0] = 1-ys[advantages[:,0]<0] 
        #the gradient change for an actions is proportional to the absolute value of the advantage
        ys = np.multiply(ys, np.abs(advantages))
        #weight by the original probabilities (this seems to actually slow down the convergence)
        #ys = np.multiply(ys, probabilities)
        
        #update the action model         
        update_model(action_model, states, ys)

    total_reward = np.sum(rewards) 
    return total_reward


#-------------------------main program-----------------------
env = gym.make('CartPole-v1')

action_model = create_action_model()
value_model = create_value_model()

rewards = []
for k in range(2000):
    reward = run_episode(env, action_model, value_model, explore=True)
    print ('Train episode {}: you survived {} steps!'.format(k, reward))
    rewards.append(reward)    
    if len(rewards)>=5 and np.min(rewards[-5:])==MAX_GAME_STEPS:
        #the algorithm tends to diverge if we do not stop it at the right time.
        #quit if some number of most recent episodes achieved max reward
        break

print ('testing...')
rewards = []
for k in range(100):
    reward = run_episode(env, action_model, value_model, explore=False)
    rewards.append(reward)
    print ('Test episode {}: you survived {} steps!'.format(k, reward))

print ('mean test reward = {}!'.format(np.mean(rewards)))

env.close()

