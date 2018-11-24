import numpy as np
import random
from collections import namedtuple, deque
import torch

device = torch.device("cpu")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        
        field_names = ["state", "full_state", "action", "full_action", "reward", "next_state", "full_next_state", "full_next_action", "done"]
        self.experience = namedtuple("Experience", field_names=field_names)
        self.seed = random.seed(seed)
    
    def add(self, agent, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        num_agents = state.shape[0]
        
        full_state = np.concatenate((state), axis=None)
        full_action = np.concatenate((action), axis=None)
        full_next_state = np.concatenate((next_state), axis=None)
        
        full_next_action = np.zeros((agent.num_agents, agent.action_size))
        #for i in range(agent.num_agents):
        #    actor_target_input = torch.from_numpy(next_state[i]).float().to(device)
        #    next_action = agent.actor_target(actor_target_input).cpu().data.numpy()
        #    next_actions[i, :] = next_action
            
        #full_next_action = np.concatenate((next_actions), axis=None)
        
        for i in range(num_agents):
            e = self.experience(state[i], full_state, action[i], full_action, reward[i], next_state[i], full_next_state, full_next_action, done[i])
            self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        full_states = torch.from_numpy(np.vstack([e.full_state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        full_actions = torch.from_numpy(np.vstack([e.full_action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        full_next_states = torch.from_numpy(np.vstack([e.full_next_state for e in experiences if e is not None])).float().to(device)
        full_next_actions = torch.from_numpy(np.vstack([e.full_next_action for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, full_states, actions, full_actions, rewards, next_states, full_next_states, full_next_actions, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)