#https://github.com/defef/reinforcement-learning-examples
import numpy as np
import random
import gym
from keras.models import Sequential
from keras.layers import Dense, Activation


ALPHA = 0.01 #learning rate in the Q value update equation
GAMMA = 0.97 #discount factor
MAX_GAME_STEPS = 500
MEMORY_SIZE = 1000 #each step is one record (s,a,r,s') in the memory 

def create_Q_model():
    model = Sequential()
    #simple Q model - input is the concatenation of the state (dim=4) 
    #and the action (dim=1)
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


#add datapoints in x to the end of q, 
#and then remove data from the beginning of q 
#to ensure that the size of q does not exceed MEMORY_SIZE
def push_to_quenue(q,x):
    if len(q)>0:
        q = np.concatenate((q, x), axis = 0)
    else:
        q = x
    q = q[-min(len(q), MEMORY_SIZE):,:]
    return q


#memory for the datapoints (s,a,r,s') implemented via queues.
def push_to_memory(experience, 
                   current_states, 
                   current_actions, 
                   current_total_rewards, 
                   next_states
                   ):
    if len(experience) == 0:
        experience = {'current_states': [], 
                      'current_actions':[], 
                      'current_total_rewards':[],
                      'next_states':[]
                    }
 
    experience['current_states'] = push_to_quenue(experience['current_states'],current_states)
    experience['current_actions'] = push_to_quenue(experience['current_actions'],current_actions)
    experience['current_total_rewards'] = push_to_quenue(experience['current_total_rewards'],current_total_rewards)
    experience['next_states'] = push_to_quenue(experience['next_states'],next_states)

    return experience


def run_train_episode(env, Q_model, exploration_rate, experience):
    current_states = []
    current_actions = []
    next_states = []
    next_max_Q_vals = []
    
    current_state = env.reset()
    #using the current state s, determine action a, act, get the immediate reward r and next state s'.
    #do not update the Q model, just record the data (s,a,r,s') for this episode 
    for k in range(MAX_GAME_STEPS):
        #env.render()
        #choose action
        predicted_Q_vals = predict_Q_values(Q_model, current_state)
        #if argmax(current_Q_vals)=0 then go left (cartpole action 0), if =1 then go right (cartpole action 1)
        current_action = np.argmax(predicted_Q_vals)
        #flip action for exploration
        if random.random() < exploration_rate:
            current_action = 0 if random.random() < 0.5 else 1
                
        current_relevant_Q_val = np.max(predicted_Q_vals)
        #take action. 
        next_state, current_reward, done, info = env.step(current_action)

        next_max_Q_val = np.max(predict_Q_values(Q_model, next_state))

        #accumulte information for the current episode
        current_states.append(current_state)
        current_actions.append(current_action)
        next_states.append(next_state)
        next_max_Q_vals.append(next_max_Q_val)
        
        current_state = next_state        
        
        if done:
            break

    #using immediate reward of 1 in the Q-value update does not seem to work.
    #setting the reward from the very last action in the episode to zero improves
    #things, but convergence is quote slow.  
    #setting the immediate reward equal to the total cumulative reward works best. 
    total_reward = len(current_actions)
    current_total_rewards = range(len(current_actions),0,-1) #assuming one step has an immediate reward of 1

    #add the datapoints from the current episode into memory
    experience = push_to_memory(
        experience, 
        np.array(current_states), 
        np.array(current_actions).reshape(len(current_actions),1),
        np.array(current_total_rewards).reshape(len(current_actions),1),
        np.array(next_states),
    )
    
    #retrieve the entire memory of (s,a,r,s') records
    current_states = experience['current_states']
    current_actions = experience['current_actions']
    current_total_rewards = experience['current_total_rewards'] 
    next_states = experience['next_states']

    #get the Q_vals for the state s and action a in the memory    
    X = np.concatenate((np.array(current_states), np.array(current_actions)), axis=1)
    current_Q_vals = Q_model.predict(X, verbose=0).reshape((len(X),1))
    
    #get the max Q_vals (across all actions) for the "next state" s' taken from the memory 
    X0 = np.concatenate(( np.array(next_states), np.zeros((len(next_states),1))), axis=1)
    X1 = np.concatenate(( np.array(next_states), np.ones((len(next_states),1))), axis=1)    
    next_Q0 = Q_model.predict(X0, verbose=0)
    next_Q1 = Q_model.predict(X1, verbose=0)
    next_max_Q_vals = np.concatenate((next_Q0, next_Q1), axis=1).max(axis=1).reshape((len(next_Q0),1))
    
    #calculate expected Q_vals and use them as the target for updating the Q_model
    expected_Q_vals = current_Q_vals + ALPHA*(current_total_rewards + GAMMA*next_max_Q_vals-current_Q_vals)    
    update_model(Q_model, X, expected_Q_vals)

    return total_reward, experience


#-------------------------main program-----------------------
env = gym.make('CartPole-v1')
  
Q_model = create_Q_model()
 
training_episodes_count = 500
experience = []
for k in range(training_episodes_count):
    exploration_rate = 0.9*(1.0-k/(2.*training_episodes_count)) #start with 0.9 and gradually reduce to 0.45
    reward, experience = run_train_episode(env, Q_model, exploration_rate, experience)
    print ('Train episode {}: you survived {} steps!'.format(k, reward))

print ('testing...')
rewards = []
for k in range(100):
    reward = run_test_episode(env, Q_model)
    rewards.append(reward)
    print ('Test episode {}: you survived {} steps!'.format(k, reward))

print ('mean test reward = {}!'.format(np.mean(rewards)))

env.close()






