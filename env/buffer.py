import random
import numpy as np

class Buffer:
    def __init__(self, buffer_size=100, batch_size=64):
        """
        A simple replay buffer for storing transitions.

        :param buffer_size: Maximum number of transitions the buffer can hold.
        :param batch_size: Number of transitions to sample in each batch.
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = []
        self.position = 0

    def collect(self, state, action, reward, next_state, done):
        """
        Store a single transition in the buffer.

        :param state: previous state.
        :param action: Action taken.
        :param reward: Reward received for action.
        :param next_state: current new state after the action.
        :param done: Boolean indicating if the episode was done after the Action was given.
        """
        transition = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition)
        # else:
            # print("WARNING: BUFFER SIZE FULL NOT STORING EXTRA INFORMATION")
        else:
            print("BUFFER FLOODDED!!", len(self.buffer))
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.buffer_size

    def generate_batches(self, num_batches=1):
        """
        Generate batches of transitions for training, yielding them based on the number of batches requested.
        First shuffles the buffer, then loops the buffer size // batch size number of times to yield batches.

        :param num_batches: Number of batches to yield.
        :return: Yields a batch of transitions as (states, actions, rewards, next_states, dones).
        """
        # Check if there's enough data for at least one batch
        if len(self.buffer) < self.batch_size:
            raise ValueError("Not enough data in buffer to generate a batch.")
        
        # Shuffle the buffer before generating batches
        random.shuffle(self.buffer)

        # Loop through the buffer in chunks of batch_size
        for _ in range(num_batches):
            # Create a batch by slicing the shuffled buffer
            batch = self.buffer[:self.batch_size]
            self.buffer = self.buffer[self.batch_size:]  # Update buffer to remove the used batch

            # Separate the batch into states, actions, rewards, next_states, and dones
            states, actions, rewards, next_states, dones = zip(*batch)

            # Yield the batch as numpy arrays
            yield (
                np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.float32),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32),
            )

    def reset(self):
        """Clear the buffer."""
        self.buffer = []
        self.position = 0
