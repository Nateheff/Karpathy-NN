import torch
import torch.nn as nn

from helpers import *

class Tokenizer:
    def __init__(self, vocab_size):
        self.merges = {}
        self.pattern = ""
        self.special_tokens = []
        self.vocab = None
        self.vocab_size = vocab_size
        self.base_tokens = 256

    def train(self, text):
        num_merges = self.vocab_size - self.base_tokens
        tokens = text.encode('utf-8')
        tokens = list(map(int,tokens))
        ids = list(tokens)
        vocab = {idx:bytes(ids[idx]) for idx in range(self.base_tokens)}
        for i in range(num_merges):
            stats = get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = self.base_tokens + i
            ids = merge(ids, idx, pair)
            self.merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        self.vocab = vocab
    
        return ids
    
    def encode(self, text):
        tokens = list(text.encode('utf-8'))
        
        while len(tokens) >= 2:
            stats = get_stats(tokens)
            pair = max(stats, key=stats.get)
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = merge(tokens, idx, pair)
        
        return tokens
    
    def decode(self, ids):
        vocab = {idx: bytes([idx]) for idx in range(self.base_tokens)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]

        tokens = b"".join(vocab[idx] for idx in ids)
        text = tokens.decode('utf-8', errors='replace')
        return text
