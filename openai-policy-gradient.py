#https://github.com/defef/reinforcement-learning-examples
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Activation


class GymEnvironment:    
    env = None
    
    def __init__(self, name):
        import gym
        self.env = gym.make(name)
        
    def reset(self):
        return self.env.reset()

    def step(self, action):
        #self.env.render()
        next_state, reward, done, info = self.env.step(action)
        return next_state, reward, done

    def get_available_actions(self): 
        return [0,1]
   

class FlappyBirdEnvironment:    
    env = None
    game = None
    
    def __init__(self, display=False):
        from ple.games.flappybird import FlappyBird
        from ple import PLE
        
        self.game = FlappyBird()
        self.env = PLE(self.game, fps=30, display_screen=display)
        self.env.init()
    
    def get_processed_game_state(self):
        state = self.game.getGameState()
        min_vals = np.array([131.,183,31,126,39,26,-16,-7])
        max_vals = np.array([289.,427,189,292,283,192,10,284])    
        state = np.array([state[x] for x in sorted(state.keys())])
        state = (state - min_vals) / (max_vals - min_vals)
        return (state)    
        
    def reset(self):
        self.env.reset_game()
        current_state = self.get_processed_game_state()
        return current_state

    def step(self, action):
        reward = self.env.act(action)        
        next_state = self.get_processed_game_state()   
        done = self.env.game_over()
        return next_state, reward, done

    def get_available_actions(self): 
        return self.env.getActionSet()
   
   
class GenericAgent:    
    max_game_steps = None
    environment = None
    
    def __init__(self, environment, max_game_steps):
        self.max_game_steps = max_game_steps
        self.environment = environment

    def act(self, state, policy):
        action_index, action_probability = policy.choose_action(state)
        action = self.environment.get_available_actions()[action_index]
        next_state, reward, done = environment.step(action)        
        return  action_index, action_probability, reward, next_state, done

    def run_episode(self, policy):
        action_indices = []
        action_probabilities = []
        rewards = []
        current_states = []
        next_states = []
        
        current_state = environment.reset()
        for _ in range(self.max_game_steps):
            #act according to the policy
            action_index, action_probability, reward, next_state, done = self.act(current_state, policy)
            
            #accumulate results
            action_indices.append(action_index)
            action_probabilities.append(action_probability)
            rewards.append(reward)
            current_states.append(current_state)
            next_states.append(next_state)
                        
            current_state = next_state          

            if done:
                break

        return action_indices, action_probabilities, rewards, current_states, next_states


class CartPolePolicyWithValueActioModel:    
    action_model = None
    value_model = None
    explore = None
      
    def __init__(self):
        self.action_model = self._create_action_model()
        self.value_model = self._create_value_model()


    def choose_action(self, state):
        probabilities = self.action_model.predict(state.reshape(1, state.size), verbose=0)[0]
        if self.explore: 
            #explore/exploit - choose action randomly in proportion to it's probability
            rand = random.random()
            action, probability = (0,probabilities[0]) if rand<probabilities[0] else (1,probabilities[1]) 
        else:
            #exploit only
            action, probability = (0,probabilities[0]) if probabilities[0]>probabilities[1] else (1,probabilities[1])   
        return action, probability   
             
    
    def _create_action_model(self):
        model = Sequential()
        model.add(Dense(8, input_shape=(4,),  activation='sigmoid'))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='binary_crossentropy', optimizer='adam')
        print(model.summary())
        return model


    def _create_value_model(self):
        model = Sequential()
        model.add(Dense(10, input_shape=(4,),  activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        print(model.summary())
        return model
    

class CartPolePolicyWithQFunction:    
    Q_model = None
    exploration_rate = None
    
    number_of_actions = 2
    state_dim = 4
      
    def __init__(self):
        self.Q_model = self._create_Q_model()

    def predict_Q_values(self, states):        
        states = np.array(states)
        if len(states.shape) < 2:
            states = np.array([states]) 
        return self.Q_model.predict(states, verbose=0)

    def get_max_Q_values(self, states):        
        next_max_Q_vals = np.max(self.predict_Q_values(states))
        return next_max_Q_vals
    
    def choose_action(self, state):
        predicted_Q_vals = self.predict_Q_values(state)
        #if argmax(current_Q_vals)=0 then go left (cartpole action 0), if =1 then go right (cartpole action 1)
        action = np.argmax(predicted_Q_vals)
        
        #flip action for exploration        
        need_to_explore =  True if random.random() < self.exploration_rate else False
        random_action = 0 if random.random() < 0.5 else 1
        if need_to_explore:
            action = random_action            
        return action, None    
    
    def _create_Q_model(self):
        model = Sequential()
        #simple Q model - input is the concatenation of the state (dim=4) 
        #and the one-hot action representation (dim=1)
        model.add(Dense(10, input_shape=(4,),  activation='relu'))
        model.add(Dense(2, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        print(model.summary())
        return model


class FlappyBirdPolicyWithValueActioModel:    
    action_model = None
    value_model = None
    explore = None
      
    def __init__(self):
        self.action_model = self._create_action_model()
        self.value_model = self._create_value_model()

    def choose_action(self, state):
        probabilities = self.action_model.predict(state.reshape(1, state.size), verbose=0)[0]
        if self.explore: 
            #explore/exploit - choose action randomly in proportion to it's probability
            rand = random.random()
            action, probability = (0,probabilities[0]) if rand<probabilities[0] else (1,probabilities[1]) 
        else:
            #exploit only
            action, probability = (0,probabilities[0]) if probabilities[0]>probabilities[1] else (1,probabilities[1])   
        return action, probability    

    
    def _create_action_model(self):
        model = Sequential()
        model.add(Dense(8, input_shape=(8,),  activation='sigmoid'))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='binary_crossentropy', optimizer='adam')
        print(model.summary())
        return model

    def _create_value_model(self):
        model = Sequential()
        model.add(Dense(10, input_shape=(8,),  activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        print(model.summary())
        return model
    

class PolicyGradientLearner:
    reward_discount_factor = 0.99
    policy = None
    agent = None

    def __init__(self, agent, policy):
        self.agent = agent
        self.policy = policy
            
    def update_model(self, model, X, y):
        model.fit(X, y,
                batch_size=5,
                epochs=10,
                verbose=0,
                shuffle=True,
                )    

    def get_discounted_total_rewards(self, rewards):
        total_discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(xrange(0, rewards.size)):
            running_add = running_add * self.reward_discount_factor + rewards[t]
            total_discounted_rewards[t] = running_add
        return total_discounted_rewards
    
    def convert_to_1_hot(self, x):
        x = x.reshape((len(x)))
        x_1_hot = np.zeros((len(x), len(self.agent.environment.get_available_actions())))
        x_1_hot[np.arange(len(x)), x] = 1
        return x_1_hot
    
    def run_train_episode(self):
        #run an episode with exploration
        policy.explore = True        
        action_indices, action_probabilities, rewards, current_states, next_states = agent.run_episode(policy)

        #reshape and convert data structures
        states = np.reshape(current_states, (len(current_states), len(current_states[0])))
        action_indices = np.reshape(action_indices, (len(action_indices), 1))
        rewards = np.array(rewards)
        action_probabilities = np.array(action_probabilities)

        #get incurred discounted rewards         
        discounted_total_rewards = self.get_discounted_total_rewards(rewards)
        #get predicted discounted rewards         
        predicted_discounted_total_rewards = self.policy.value_model.predict(states, verbose=0).reshape(len(rewards))
        
        #update value model
        self.update_model(self.policy.value_model, states, discounted_total_rewards)
        
        #negative advantage means that the discounted reward for the action
        #was lower than expected by the value-model 
        advantages = discounted_total_rewards - predicted_discounted_total_rewards
        
        #determine in which direction to change the gradients.
        #first, identify which actions should've been picked according to the advantages.
        #if advantage is negative then we should increase the probability of the action that we did not take.
        ys = self.convert_to_1_hot(action_indices)
        advantages = np.repeat(advantages.reshape(rewards.size,1), ys.shape[1], axis=1)
        ys[advantages[:,0]<0] = 1-ys[advantages[:,0]<0] 
        #the gradient change for an actions is proportional to the absolute value of the advantage
        ys = np.multiply(ys, np.abs(advantages))
        #weight by the original probabilities (this seems to actually slow down the convergence)
        #ys = np.multiply(ys, np.repeat(action_probabilities.reshape(rewards.size,1), ys.shape[1], axis=1) )
        
        #update the action model         
        self.update_model(self.policy.action_model, states, ys)
    
        total_reward = np.sum(rewards) 
        return total_reward


    def train(self, iterations):
        rewards = []
        for k in range(iterations):
            reward = self.run_train_episode()
            if reward > -5:
                print ('Train episode {}: you survived {} steps!'.format(k, reward))
            
            rewards.append(reward)    
            if len(rewards)>=5 and np.min(rewards[-5:])==agent.max_game_steps:
                #the algorithm tends to diverge if we do not stop it at the right time.
                #quit if some number of most recent episodes achieved max reward
                break        

        return self.policy


class ReplayMemory:
    memory = []
    memory_size = None
    
    def __init__(self, memory_size):
        self.agent = agent
    
    #add datapoints in x to the end of q, 
    #and then remove data from the beginning of q 
    #to ensure that the size of q does not exceed memory_size
    def _push_to_quenue(self, x):
        self.memory = self.memory + x
        self.memory = self.memory[-np.min(len(self.memory), self.memory_size):]
            
    def push(self, x1,x2,x3,x4):
        self._push_to_quenue(zip(x1,x2,x3,x4))
                
    def get_sample(self, sample_size):
        x1,x2,x3,x4 = zip(*random.sample(self.memory, np.min([len(self.memory), sample_size])))        
        x1 = np.array(x1)
        x2 = np.array(x2).reshape(len(x2),1)
        x3 = np.array(x3).reshape(len(x3),1)
        x4 = np.array(x4)
        return x1,x2,x3,x4
        

class PolicyQWithReplayLearner:    
    ALPHA = 0.01 #learning rate in the Q value update equation
    GAMMA = 0.97 #discount factor
    MEMORY_SIZE = 500 #each step is one record (s,a,r,s') in the memory
    MEMORY_SAMPLE_SIZE = 500 
    
    policy = None
    agent = None    
    memory = None

    def __init__(self, agent, policy):
        self.agent = agent
        self.policy = policy
        self.memory = ReplayMemory(self.MEMORY_SIZE)
            
    def _update_model(self, model, X, y):
        model.fit(X, y,
                batch_size=5,
                epochs=10,
                verbose=0,
                shuffle=True,
                )   
    
    def _convert_to_1_hot(self, x):
        x = x.reshape((len(x)))
        x_1_hot = np.zeros((len(x), len(self.agent.environment.get_available_actions())))
        x_1_hot[np.arange(len(x)), x] = 1
        return x_1_hot
    
    def run_train_episode(self, experience):
        policy.exploration_rate = 0.4        
        current_action_indices, action_probabilities, current_rewards, current_states, next_states = agent.run_episode(policy)  

        #CHANGING REWARD TO SPEED UP CONVERGENCE 
        #using immediate reward of 1 in the Q-value update does not seem to work.
        #setting the reward from the very last action in the episode to zero improves
        #things, but convergence is quote slow.  
        #setting the immediate reward equal to the total cumulative reward works best.
        current_rewards = range(len(current_rewards),0,-1) 
        print ('episode reward={}'.format(len(current_rewards)))
    
        #add the datapoints from the current episode into memory
        self.memory.push(current_states, current_action_indices, current_rewards, next_states)
        
        #retrieve the entire memory of (s,a,r,s') records
        current_states, current_action_indices, current_rewards, next_states = self.memory.get_sample(self.MEMORY_SAMPLE_SIZE)
    
        #get the Q_vals for the state s and action a in the memory    
        current_Q_vals = policy.predict_Q_values(current_states)
        
        
        #current_states_and_actions = np.concatenate((np.array(current_states), np.array(current_actions)), axis=1)
        
        #extract q-vals that correspond only to the actions we are interestd in
        #current_Q_vals = current_Q_vals[np.arange(len(current_Q_vals)), current_actions.reshape((len(current_actions),))]
        #current_Q_vals = current_Q_vals.reshape((len(current_Q_vals),1))
        
        next_Q_vals = policy.predict_Q_values(next_states)        
#        next_max_Q_vals = next_Q_vals.max(axis=1).reshape((len(next_Q_vals),1))

        next_max_Q_vals = np.repeat(next_Q_vals.max(axis=1).reshape((len(next_Q_vals),1)), next_Q_vals.shape[1], axis=1 )
        
        current_action_indices_one_hot = self._convert_to_1_hot(current_action_indices)                
        
        #calculate expected Q_vals and use them as the target for updating the Q_model
        expected_Q_vals = current_Q_vals + self.ALPHA*(current_rewards + self.GAMMA*next_max_Q_vals-current_Q_vals) 
        
        #expected Q-values are only valid for the actions that we took. use current q-values for actions that we did not take
        expected_Q_vals = expected_Q_vals*current_action_indices_one_hot + current_Q_vals*(1-current_action_indices_one_hot) 
           
        #REDO: IT IS PROBABLY BETTER TO HAVE A CUSTOM LOSS AND UPDATE ONLY FOR THE ACTIONS THAT WE'VE TAKEN 
        self._update_model(self.policy.Q_model, current_states, expected_Q_vals)
        
        total_reward = np.sum(current_rewards)
        return total_reward, experience


    def train(self, iterations):
        experience = []
        for k in range(iterations):
            self.exploration_rate = 0.9*(1.0-k/(2.*iterations)) 
            reward, experience = self.run_train_episode(experience)
            print ('Train episode {}:expl rate={:.2f}, reward={}'.format(k, self.exploration_rate, reward))
            
            #if k%10000==0:
            #    Q_model.save('./action_model.mdl')        

        return self.policy



#-------------------------main program-----------------------

##CartPole-v0 environment and policy gradient
#environment = GymEnvironment('CartPole-v0')
#policy = CartPolePolicyWithValueActioModel()
#agent = GenericAgent(environment, max_game_steps=200)
#learner = PolicyGradientLearner(agent, policy)

# #FalppyBird environment and policy gradient - DOES NOT CONVERGE!
# environment = FlappyBirdEnvironment(display=False)
# policy = FlappyBirdPolicyWithValueActioModel()
# agent = GenericAgent(environment, max_game_steps=200)
# learner = PolicyGradientLearner(agent, policy)

#CartPole-v0 environment and Q-learning
environment = GymEnvironment('CartPole-v0')
policy = CartPolePolicyWithQFunction()
agent = GenericAgent(environment, max_game_steps=200)
learner = PolicyQWithReplayLearner(agent, policy)

#train policy
policy = learner.train(2000)

#test policy
policy.explore = False
print ('testing...')
all_steps_counts = []
for k in range(10):
    actions, _, _, _, _ = agent.run_episode(policy)
    steps_count = len(actions)
    print ('Test episode {}: you survived {} steps!'.format(k, steps_count))
    all_steps_counts.append(steps_count)
 
print ('mean test reward = {}!'.format(np.mean(all_steps_counts)))
