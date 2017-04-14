#https://github.com/defef/reinforcement-learning-examples
import numpy as np
import random
import gym
from keras.models import Sequential
from keras.layers import Dense, Activation


ALPHA = 0.1 #learning rate
GAMMA = 0.97 #discount factor
MAX_GAME_STEPS = 500

def create_Q_model():
    model = Sequential()
    #simple Q model - input is the concatenation of the state (dim=4) 
    #and the one-hot action representation (dim=1)
    model.add(Dense(10, input_shape=(4+1,),  activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())
    return model


def update_model(model, X, y):
    model.fit(X, y,
            batch_size=5,
            epochs=10,
            verbose=0,
            shuffle=True,
            )    


def predict_Q_values(Q_model, state):
    actions = [[0], [1]]  
    X = np.concatenate((np.array([state]*len(actions)), actions), axis=1) 
    Q_vals = Q_model.predict(X, verbose=0)    
    return Q_vals
        

def run_test_episode(env, Q_model):
    rewards = []
    state = env.reset()
    for _ in range(MAX_GAME_STEPS):
        env.render()
        current_Q_vals = predict_Q_values(Q_model, state)
        state, reward, done, info = env.step(np.argmax(current_Q_vals))
        
        rewards.append(reward)
        if done:
            break
    
    total_reward = np.sum(rewards) 
    return total_reward


def run_train_episode(env, Q_model, exploration_rate):
    current_states = []
    current_actions = []
    next_states = []
    current_relevant_Q_vals = [] #current Q vals for taken actions
    next_max_Q_vals = []
    
    current_state = env.reset()
    for k in range(MAX_GAME_STEPS):
        #env.render()
        #choose action
        predicted_Q_vals = predict_Q_values(Q_model, current_state)
        #if argmax(current_Q_vals)=0 then go left (cartpole action 0), if =1 then go right (cartpole action 1)
        current_action = np.argmax(predicted_Q_vals)
        #flip action to achieve exploration
        if random.random() < exploration_rate:
            current_action = 0 if random.random() < 0.5 else 1
                
        current_relevant_Q_val = np.max(predicted_Q_vals)
        #take action 
        next_state, current_reward, done, info = env.step(current_action)

        next_max_Q_val = np.max(predict_Q_values(Q_model, next_state))

        current_states.append(current_state)
        current_actions.append(current_action)
        next_states.append(next_state)
        current_relevant_Q_vals.append(current_relevant_Q_val)
        next_max_Q_vals.append(next_max_Q_val)
        
        current_state = next_state        
        
        if done:
            break

    total_reward = len(current_actions)
    #using immediate reward of 1 in the Q-value update does not seem to work.
    #setting the reward from the very last action in the episode to zero improves
    #things, but convergence is quote slow.  
    #setting the immediate reward equal to the total cumulative reward works best. 
    current_total_rewards = range(len(current_actions),0,-1) 
    next_max_Q_vals = np.array(next_max_Q_vals)    
    
    X = np.concatenate((np.array(current_states), np.array(current_actions).reshape(len(current_actions),1)), axis=1)
    expected_Q_vals = current_relevant_Q_vals + ALPHA*(current_total_rewards + GAMMA*next_max_Q_vals-current_relevant_Q_vals)
    update_model(Q_model, X, expected_Q_vals)

    return total_reward


#-------------------------main program-----------------------
env = gym.make('CartPole-v1')

Q_model = create_Q_model() 
training_episodes_count = 1000
for k in range(training_episodes_count):
    exploration_rate = 0.9*(1.0-k/(2.*training_episodes_count)) #start with 0.9 and gradually reduce to 0.45
    reward = run_train_episode(env, Q_model, exploration_rate)
    print ('Train episode {}: you survived {} steps!'.format(k, reward))
 
print ('testing...')
rewards = []
for k in range(100):
    reward = run_test_episode(env, Q_model)
    rewards.append(reward)
    print ('Test episode {}: you survived {} steps!'.format(k, reward))

print ('mean test reward = {}!'.format(np.mean(rewards)))

env.close()

