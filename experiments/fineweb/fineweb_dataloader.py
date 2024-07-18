import numpy as np
import torch
import os
import tiktoken

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "data/edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

    def set_current_position(self, tokens):
        print()
        print('[process_rank: {self.process_rank}] CALCULATING CURRENT POSITION WITHIN DATALOADER')
        # set current position according to process_rank (offset so that processes don't overlap)
        tokens = tokens + self.B * self.T * self.process_rank
        while True:
            print(f'[process_rank: {self.process_rank}] At token position: {tokens:,} in shard: {self.current_shard}')
            if tokens + (self.B * self.T * self.process_rank + 1) < len(self.tokens):
                print(f'[process_rank: {self.process_rank}] Token position is within current shard (shard: {self.current_shard})')
                print(f'[process_rank: {self.process_rank}] Setting current position: {tokens + self.B * self.T * self.process_rank:,}')
                self.current_position = tokens
                break
            else:
                print(f'[process_rank: {self.process_rank}] Do not fit (i.e., overflow) in current shard (shard: {self.current_shard}), moving to next shard')
                tokens -= len(self.tokens)
                self.current_shard = (self.current_shard + 1) % len(self.shards)
                self.tokens = load_tokens(self.shards[self.current_shard])