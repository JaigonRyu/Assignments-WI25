import numpy as np
import random
from collections import defaultdict
from src.racetrack import RaceTrack

class MonteCarloControl:
    """
    Monte Carlo Control with Weighted Importance Sampling for off-policy learning.
    
    This class implements the off-policy every-visit Monte Carlo Control algorithm
    using weighted importance sampling to estimate the optimal policy for a given
    environment.
    """
    def __init__(self, env: RaceTrack, gamma: float = 1.0, epsilon: float = 0.1, Q0: float = 0.0, max_episode_size : int = 1000):
        """
        Initialize the Monte Carlo Control object. 

        Q, C, and policies are defaultdicts that have keys representing environment states.  
        Defaultdicts (search up the docs!) allow you to set a sensible default value 
        for the case of Q[new state never visited before] (and likewise with C/policies).  
        

        Hints: 
        - Q/C/*_policy should be defaultdicts where the key is the state
        - each value in the dict is a numpy vector where position is indexed by action
        - That is, these variables are setup like Q[state][action]
        - state key will be the numpy state vector cast to string (dicts require hashable keys)
        - Q should default to Q0, C should default to 0
        - *_policy should default to equiprobable (random uniform) actions
        - store everything as a class attribute:
            - self.env, self.gamma, self.Q, etc...

        Args:
            env (racetrack): The environment in which the agent operates.
            gamma (float): The discount factor.
            Q0 (float): the initial Q values for all states (e.g. optimistic initialization)
            max_episode_size (int): cutoff to prevent running forever during MC
        
        Returns: none, stores data as class attributes
        """
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q0 = Q0
        self.max_episode_size = max_episode_size

        self.Q = defaultdict(lambda: np.full(self.env.n_actions, Q0))
        self.target_policy = defaultdict(lambda: np.ones(self.env.n_actions) / self.env.n_actions)
        self.behavior_policy = defaultdict(lambda: np.ones(self.env.n_actions) / self.env.n_actions)
        self.C = defaultdict(lambda: np.zeros(self.env.n_actions))


    def create_target_greedy_policy(self):
        """
        Loop through all states in the self.Q dictionary. 
        1. determine the greedy policy for that state
        2. create a probability vector that is all 0s except for the greedy action where it is 1
        3. store that probability vector in self.target_policy[state]

        Args: none
        Returns: none, stores new policy in self.target_policy
        """
        # Your code here

        for state in self.Q:

            state = str(state)

            best_action = np.argmax(self.Q[state])

            prob_vector = np.zeros_like(self.Q[state])

            prob_vector[best_action] = 1.0

            self.target_policy[state] = prob_vector



    def create_behavior_egreedy_policy(self):
        """
        Loop through all states in the self.target_policy dictionary. 
        Using that greedy probability vector, and self.epsilon, 
        calculate the epsilon greedy behavior probability vector and store it in self.behavior_policy[state]
        
        Args: none
        Returns: none, stores new policy in self.target_policy
        """
        # Your code here

        num = self.env.n_actions

        for state in self.target_policy:

            state = str(state)
            
            prob_vector = np.ones(num) * (self.epsilon / num)

            greedy = np.argmax(self.target_policy[state])

            prob_vector[greedy] += (1-self.epsilon)

            self.behavior_policy[state] = prob_vector


        
    def egreedy_selection(self, state):
        """
        Select an action proportional to the probabilities of epsilon-greedy encoded in self.behavior_policy
        HINT: 
        - check out https://www.w3schools.com/python/ref_random_choices.asp
        - note that random_choices returns a numpy array, you want a single int
        - make sure you are using the probabilities encoded in self.behavior_policy 

        Args: state (string): the current state in which to choose an action
        Returns: action (int): an action index between 0 and self.env.n_actions
        """
        # Your code here

        action = random.choices(range(self.env.n_actions), weights=self.behavior_policy[state])[0]

        #print(action)

        return int(action)

    def generate_egreedy_episode(self):
        """
        Generate an episode using the epsilon-greedy behavior policy. Will not go longer than self.max_episode_size
        
        Hints: 
        - need to setup and use self.env methods and attributes
        - use self.egreedy_selection() above as a helper function
        - use the behavior e-greedy policy attribute aleady calculated (do not update policy here!)
        
        Returns:
            list: The generated episode, which is a list of (state, action, reward) tuples.
        """
        # Your code here

        episode = []

        state = self.env.reset()
        state = str(state)

        for _ in range(self.max_episode_size):

            action = self.egreedy_selection(state)
            
            reward = self.env.take_action(int(action))
            next_state = tuple(self.env.get_state())
            done = self.env.is_terminal_state()

            next_state = str(next_state)

            episode.append((state, action, reward))

            if done:
                break
            
            state = next_state

            

        return episode
        
    
    def generate_greedy_episode(self):
        """
        Generate an episode using the greedy target policy. Will not go longer than self.max_episode_size
        Note: this function is not used during learning, its only for evaluating the target policy
        
        Hints: 
        - need to setup and use self.env methods and attributes
        - use the greedy policy attribute aleady calculated (do not update policy here!)

        Returns:
            list: The generated episode, which is a list of (state, action, reward) tuples.
        """
        # Your code here

        episode = []

        state = self.env.reset()
        state = str(state)

        for i in range(self.max_episode_size):

            action = np.argmax(self.target_policy[state])

            
            reward = self.env.take_action(int(action))
            next_state = tuple(self.env.get_state())
            done = self.env.is_terminal_state()

            next_state = str(next_state)

            episode.append((state, action, reward))

            

            state = next_state

            if done == True:
                break
        

        #print(episode)
        return episode
    
    def update_offpolicy(self, episode):
        """
        Update the Q-values using every visit weighted importance sampling. 
        See Figure 5.9, p. 134 of Sutton and Barto 2nd ed.
        
        Args: episode (list): An episode generated by the behavior policy; a list of (state, action, reward) tuples.
        Returns: none
        """
        
        G = 0
        W = 1

        for t in reversed(range(len(episode))):

            state, action, reward = episode[t]
            state = str(state)

            G = self.gamma * G + reward

            self.C[state][action] = self.C[state][action] + W
            self.Q[state][action] = self.Q[state][action] + (W/self.C[state][action]) * (G - self.Q[state][action])

            
           
            
            if action != np.argmax(self.target_policy[state]):  # If action is not optimal, break
                break

            
            W = W * 1.0/self.behavior_policy[state][action]

        

            self.create_target_greedy_policy()
            
                       
        

    def update_onpolicy(self, episode):
        """
        Update the Q-values using first visit epsilon-greedy. 
        See Figure 5.6, p. 127 of Sutton and Barto 2nd ed.
        
        Args: episode (list): An episode generated by the behavior policy; a list of (state, action, reward) tuples.
        Returns: none
        """
        # Your code here

        G = 0
        visited_states = set()  # Track first-visit states

        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            state = str(state)

            G = self.gamma * G + reward  # Compute return

            if (state, action) not in visited_states:
                visited_states.add((state, action))

                self.Q[state][action] += (1 / (self.C[state][action] + 1)) * (G - self.Q[state][action])
                self.C[state][action] += 1

                # Update policy
                self.create_target_greedy_policy()
                self.create_behavior_egreedy_policy()


    def train_offpolicy(self, num_episodes):
        """
        Train the agent over a specified number of episodes.
        
        Args:
            num_episodes (int): The number of episodes to train the agent.
        """
        for _ in range(num_episodes):
            episode = self.generate_egreedy_episode()
            self.update_offpolicy(episode)

   


    def get_greedy_policy(self):
        """
        Retrieve the learned target policy in the form of an action index per state
        
        Returns:
            dict: The learned target policy.
        """
        policy = {}
        for state, actions in self.Q.items():
            policy[state] = np.argmax(actions)
        return policy