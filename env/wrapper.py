import gymnasium as gym
import numpy as np
from env.dreamer import Dreamer
from env.buffer import Buffer
from gymnasium.core import ObsType
import torch

class DreamWrapper(gym.Wrapper):
    def __init__(self, env, n_future_steps:int=1, n_steps:int = 512, n_steps_dreamer:int = 512, eval=False, dreamer_save_path='runs'):
        '''
        n_future_steps: number of future predictions by dreamer
        n_steps: update dreamer after these many step() calls. Also equal to buffer length
        '''
        super(DreamWrapper, self).__init__(env)
        self.dreamer = Dreamer(env, n_future_steps, env.action_space.shape[0] ,env.observation_space.shape[0], dreamer_save_path)
        # TODO: Check for action and observation limits in dreamer, 
        # e.g. if actions are bounded -1 to 1, dreamer policy should not have unbounded output
        self.n_future_steps = n_future_steps
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim + self.n_future_steps * self.state_dim,), dtype=np.float32
        )
        self.n_steps = n_steps
        self.buffer = Buffer(buffer_size=n_steps, batch_size=128)
        self.eval = eval
        self._seed = 47
        self.counter = 0

    def reset(self, **kwargs):
        # print(kwargs)
        if "seed" in kwargs:
            print("seeded externally")
            state, info = super().reset(**kwargs)
        else:
            state, info = super().reset(seed=self._seed)

        future_predictions = self.dreamer(state)
       
        self.state = state
       
        return np.concatenate([state, future_predictions]), info
        
    def step(self, action):
        # # TODO: store history of observations
        # # collect data for training dreamer
        # self.buffer.collect(self.state, action)
        # buff_curr_state = se
        # #train dreamer
        if not self.eval:
            if self.counter == self.n_steps:
                for state_batch, action_batch, reward_batch, next_state_batch, done_batch in self.buffer.generate_batches():
                    self.dreamer.update(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
                self.buffer.reset()
                self.counter=0
            
        next_state, reward, done, truncated, info = self.env.step(action)
      
        future_predictions = self.dreamer(next_state)
        if not self.eval:
            self.buffer.collect(self.state, action, reward, next_state, done)
            self.counter += 1

        self.state = next_state
        return np.concatenate([next_state, future_predictions]), reward, done, truncated, info
    
    # def _get_augmented_observation(self, state):
    #     augmented_observation = [state]
        
    #     for _ in range(self.n_future_steps):
    #         future_state = self.model(torch.tensor(state).float(), torch.zeros(self.action_dim).float()).detach().numpy()
    #         augmented_observation.append(future_state)
    #         state = future_state  # Update state for the next step
        
    #     # Flatten the list of future states into one single observation
    #     return np.concatenate(augmented_observation, axis=-1)

