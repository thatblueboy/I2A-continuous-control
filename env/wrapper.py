import gymnasium as gym
import numpy as np
from env.dreamer import Dreamer
from env.buffer import Buffer
from gymnasium.core import ObsType
import torch
import logging

class DreamWrapper(gym.Wrapper):
    def __init__(self, env, history_len:int=0, 
                 n_future_steps:int=2, 
                 n_steps:int = 512, 
                 n_steps_dreamer:int = 512, 
                 dreamer_batch_size:int = 64,
                 eval:bool=False, 
                 policy_hidden_layers:list=[128, 64],
                 dynamics_hidden_layers:list=[128, 64],
                 dreamer_save_path='runs'):
        '''
        n_future_steps: number of future predictions by dreamer
        n_steps: update dreamer after these many step() calls. Also equal to buffer length
        '''
        super(DreamWrapper, self).__init__(env)

        assert isinstance(n_future_steps, int), "future steps must be integer"
        self.n_future_steps = n_future_steps

        if self.n_future_steps != 0:
            self.dreamer = Dreamer(env, 
                                   n_future_steps, 
                                   env.action_space.shape[0] ,
                                   env.observation_space.shape[0], 
                                   policy_hidden_layers,
                                   dynamics_hidden_layers,
                                   dreamer_save_path)
            self.n_steps_dreamer = n_steps #update dreamer using same policy
            self.buffer = Buffer(buffer_size=n_steps_dreamer, batch_size=dreamer_batch_size)
            self.dreamer_epochs = 5
            self.dreaming = True
            print("Dreaming:", self.dreaming)
            self.reset = self.dream_reset
            self.step = self.dream_step
        else:
            self.dreaming = False
            print("Dreaming:", self.dreaming)
            self.reset = self.dream_less_reset
            self.step = self.dream_less_step

        # TODO: Check for action and observation limits in dreamer, 
        # e.g. if actions are bounded -1 to 1, dreamer policy should not have unbounded output
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=((history_len + 1 +self.n_future_steps) * self.obs_dim ,), dtype=np.float32
        )
        self.n_steps = n_steps
        self.eval = eval
        self.counter = 0
        print(history_len)
        self.obs_buf_size = history_len+1
        self.obs_buffer = np.zeros((self.obs_buf_size*self.obs_dim))

    def dream_less_reset(self, **kwargs):
        state, info = super().reset(**kwargs)
        self.state = state
        self.obs_buffer = np.repeat(state, self.obs_buf_size)
        # print("shape", self.obs_buffer.shape)
        return self.obs_buffer, info
    
    def dream_less_step(self, action):
        next_state, reward, done, truncated, info = self.env.step(action)
        self.state = next_state
        self.obs_buffer = np.concatenate([self.obs_buffer[:-self.obs_dim], next_state])
        # print(self.obs_buffer.shape)
        return self.obs_buffer, reward, done, truncated, info

    def dream_reset(self, **kwargs):
        # print(kwargs)
        state, info = super().reset(**kwargs)

        future_predictions = self.dreamer(state)
       
        self.state = state
        self.obs_buffer = np.repeat(state, self.obs_buf_size)
        # print("shape", self.obs_buffer.shape)
        return np.concatenate([self.obs_buffer, future_predictions]), info
        
    def dream_step(self, action):
        
        if not self.eval:
            if self.counter == self.n_steps_dreamer: #update first, call step later so that policy is the first to be updated
                # logging.debug("Updating dreamer!")
                for i in range(self.dreamer_epochs):
                    for state_batch, action_batch, reward_batch, next_state_batch, done_batch in self.buffer.generate_batches():
                        self.dreamer.update(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
                self.buffer.reset()
                self.counter=0
   
        next_state, reward, done, truncated, info = self.env.step(action)
      
        future_predictions = self.dreamer(next_state)
       
        if not self.eval:
            self.counter += 1
            self.buffer.collect(self.state, action, reward, next_state, done)
            # logging.debug('SARS collected for dreamer these many times: %d', self.counter)

        self.state = next_state
        self.obs_buffer = np.concatenate([self.obs_buffer[:-self.obs_dim], next_state])
            
        # print(self.obs_buffer.shape)
        return np.concatenate([self.obs_buffer, future_predictions]), reward, done, truncated, info

    # def _get_augmented_observation(self, state):
    #     augmented_observation = [state]
        
    #     for _ in range(self.n_future_steps):
    #         future_state = self.model(torch.tensor(state).float(), torch.zeros(self.action_dim).float()).detach().numpy()
    #         augmented_observation.append(future_state)
    #         state = future_state  # Update state for the next step
        
    #     # Flatten the list of future states into one single observation
    #     return np.concatenate(augmented_observation, axis=-1)

