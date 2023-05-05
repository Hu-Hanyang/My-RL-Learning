import random
import numpy as np

"""
The class ReplayBuffer1() are from https://github.com/dxyang/DQN_pytorch/blob/master/utils/replay_buffer.py.
"""

def sample_n_unique(sampling_f, n):
    """
    Helper funtion. Given the function 'sampling_f' that returns
    comparable objects, sample n such unique objects.
    """
    res = []
    while len(res) < n:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res

class ReplayBuffer1():
    def __init__(self, size, frame_history_len):
        """
        Parameters
        size: int
            The maximum number of transitions to store in the replay buffer. When
            the buffer overflows, the old memories are dropped.
        frame_history_len: int
            The number of memories to be retried for each observation. 
            This value could be more than 1 becauze of the MDP features.
        """
        self.size = size
        self.fram_history_len = frame_history_len

        self.next_idx = 0
        self.num_in_buffer = 0

        self.obs = None
        self.action = None
        self.reward = None
        self.done = None

    def can_sample(self, batch_size):
        # return True if there are 'batch_size' different transitions can be sampled from the buffer
        return batch_size + 1 <= self.num_in_buffer
    
    def _encode_observation(self, idx):
        end_idx  = idx + 1  
        start_idx = end_idx - self.fram_history_len  
        if len(self.obs.shape) == 2: 
            return self.obs[end_idx-1]
        if start_idx < 0 and self.num_in_buffer != self.size:
            start_idx = 0
        for idx in range(start_idx, end_idx-1):
            if self.done[idx % self.size]:  # start from not done state
                start_idx = idx + 1
        missing_contex = self.fram_history_len - (end_idx - start_idx)
        if start_idx < 0 or missing_contex > 0:
            frames = [np.zeros_like(self.obs[0]) for _ in range(missing_contex)]
            for idx in range(start_idx, end_idx):
                frames.append(self.obs[idx % self.size])
            return np.concatenate(frames, 0) 
        else:
            img_h, img_w = self.obs.shape[2], self.obs.shape[3]
            return self.obs[start_idx:end_idx].reshape(-1, img_h, img_w)
    
    def _encode_sample(self, idxes):
        obs_batch = np.concatenate([self._encode_observation(idx)[None] for idx in idxes], 0)
        act_batch = self.action[idxes]
        rew_batch = self.reward[idxes]
        next_obs_batch = np.concatenate([self._encode_observation(idx+1)[None] for idx in idxes], 0)
        done_mask = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)
        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask
    
    def sample(self, batch_size):
        """
        Parameters:
        batch_size: int
            The number of transitions to be sampled.
        Returns:
        obs_batch: np.array
            Shape: (batch_size, img_c * frame_history_len, img_h, img_w)
            Dtype: np.uint8
        act_batch: np.array
            Shape: (batch_size, ), dtype: np.int32
        rew_batch: np.array
            Shape: (batch_size, ), dtype: np.float32
        next_obs_batch: np.array
            Shape: (batch_size, img_c * frame_history_len, img_h, img_w)
            Dtype: np.uint8
        done_mask: np.array
            Shape: (batch_size, )
            Dtype: np.float32
        """
        assert self.can_sample(batch_size)
        idxes = sample_n_unique(lambda: random.randint(0, self.num_in_buffer-2), batch_size)
        return self._encode_sample(idxes)

    def encode_recent_observation(self):
        """ Return the most recent 'frame_history_len' frames.
        Returns:
        observation: np.array
            Shape: (img_c * frame_history_len, img_h, img_w)
            Dtype: np.uint8, where observation[i*img_c: (i+1)*img_c, :, :],
            encodes frame at time 't-frame_history_len+i'.
        """
        assert self.num_in_buffer > 0
        return self._encode_observation((self.next_idx-1) % self.size)
    
    def store_frame(self, frame):
        """
        Store a single frame in the buffer at the next available index, 
        overwrite the old frames if necessary.
        Parameters:
        frame: np.array
            Shape: (img_h, img_w, img_c)
            Dtype: np.uint8
        Returns:
        idx: int
            Index at which the frame is stored.
        """
        if len(frame.shape) > 1:
            frame = frame.transpose(2, 0, 1)  # from (img_h, img_w, img_c) to (img_c, img_h, img_w)
            
        if self.obs is None:
            self.obs = np.empty([self.size] + list(frame.shape), dtype=np.uint8)  # shape: (self.size, frame.shape)
            self.action = np.empty([self.size], dtype=np.int32)  
            self.reward = np.empty([self.size], dtype=np.float32)
            self.done = np.empty([self.size], dtype=np.bool)
        self.obs[self.next_idx] = frame

        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer+1)

        return ret

    def store_effect(self, idx, action, reward, done):
        """
        Store effects of action taken after obeserving frame stored
        at index idx. The reason `store_frame` and `store_effect` is broken
        up into two functions is so that once can call `encode_recent_observation`
        in between. 
        Parameters:
        idx: int
            The index in buffer of recently observed frame (returned by 'store_frame').
        action: int
            The action that was performed upon observing this frame.
        reward: float
            The reward that was received when the actions was performed.
        done: bool
            True if episode was finished after performing that action.
        """
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx] = done


class ReplayBuffer():
    def __init__(self, size):
        """
        Basic replay buffer 
        Parameter:
        size: int
            The maximum number of transitions (s,a,r,s,done) to be stored in this buffer.
            When the buffer overflows, the old transitions will be dropped.
        """
        self.storage = []
        self.size = size 
        self.ptr = 0

    def push(self, data):
        # data is in the shape of (s, a, r, s, done)
        if len(self.storage) == self.size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.size
        else:
            self.storage.append(data)
        
    def sample(self, batch_size):
        """
        Sample a batch size of transitions
        Parameters:
        batch_size: int
            The number of transitions to be sampled.
        Return:
        states: np.array
            The batch size of states (observations).
        actions: np.array
            The batch size of actions.
        rewards: np.array
            The batch size of rewards after executing the action.
        next_states: np.array
            The batch size of next states (observations).
        dones: np.array
            The result of executing the action, 1 means finished.
        """
        indexes = np.random.randint(0, len(self.storage), size=batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for index in indexes:
            s, a, r, next_s, d = self.storage[index]
            states.append(s)  # np.array(s, copy=False)
            actions.append(a)  # np.array(a, copy=False)
            rewards.append(r)  # np.array(r, copy=False)
            next_states.append(next_s)  # np.array(next_s, copy=False)
            dones.append(d)  # np.array(d, copy=False)
        
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)