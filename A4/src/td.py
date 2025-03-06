import numpy as np
from tqdm import tqdm
from .discretize import quantize_state, quantize_action
from dm_control.rl.control import Environment

# Initialize the Q-table with dimensions corresponding to each discretized state variable.
def initialize_q_table(state_bins: dict, action_bins: list) -> np.ndarray:
    """
    Initialize the Q-table with dimensions corresponding to each discretized state variable.

    Args:
        state_bins (dict): The discretized bins for each state variable.
        action_bins (list): The discrete actions available.

    Returns:
        np.ndarray: A Q-table initialized to zeros with dimensions matching the state and action space.
    """
    # TODO: Implement this function
    
    state_shape = []
    for state in state_bins.values():
        for bins in state:
            state_shape.append(len(bins))

    state_shape = tuple(state_shape)


    q_table = np.zeros((*state_shape, len(action_bins)))

    return q_table
    

# TD Learning algorithm
def td_learning(env: Environment, num_episodes: int, alpha: float, gamma: float, epsilon: float, state_bins: dict, action_bins: list, q_table:np.ndarray=None) -> tuple:
    """
    TD Learning algorithm for the given environment.

    Args:
        env (Environment): The environment to train on.
        num_episodes (int): The number of episodes to train.
        alpha (float): The learning rate. 
        gamma (float): The discount factor.
        epsilon (float): The exploration rate.
        state_bins (dict): The discretized bins for each state variable.
        action_bins (list): The discrete actions available.
        q_table (np.ndarray): The Q-table to start with. If None, initialize a new Q-table.

    Returns:
        tuple: The trained Q-table and the list of total rewards per episode.
    """
    if q_table is None:
        q_table = initialize_q_table(state_bins, action_bins)
        
    rewards = []

    for episode in tqdm(range(num_episodes), desc="Training Episodes"):
        # reset env
        time_step = env.reset()
        
        # run the episode
            # select action
            # take action
            # quantize the next state
            # update Q-table
            # if it is the last timestep, break
        # keep track of the reward

        state = time_step.observation
        state = quantize_state(state, state_bins)

        tot_reward = 0

        #action
        if np.random.rand() < epsilon:

            action = np.random.choice(len(action_bins))
            quantized_action = quantize_action(action, action_bins)
            
        else:

            action = np.argmax(q_table[state])
            quantized_action = quantize_action(action, action_bins)

        while not time_step.last(): 

            time_step = env.step(quantized_action)


            reward = time_step.reward

            next_state_cont = time_step.observation
            next_state_discr = quantize_state(next_state_cont, state_bins)
           
            #action
            if np.random.rand() < epsilon:

                next_action = np.random.choice(len(action_bins))
                quantized_next_action = quantize_action(next_action, action_bins)
            
            else:

                next_action = np.argmax(q_table[next_state_discr])
                quantized_next_action = quantize_action(next_action, action_bins)

            q_table[state][quantized_action] +=  alpha * (reward  + gamma * q_table[next_state_discr][quantized_next_action] - q_table[state][quantized_action])
            
            state = next_state_discr
            quantized_action = quantized_next_action
            tot_reward += reward

        rewards.append(tot_reward)

    return q_table, rewards


def greedy_policy(q_table: np.ndarray) -> callable:
    """
    Define a greedy policy based on the Q-table.    

    Args:
        q_table (np.ndarray): The Q-table from which to derive the policy.

    Returns:
        callable: A function that takes a state and returns the best action. 
    """
    def policy(state):
        
       
        return np.argmax(q_table[state])
    return policy