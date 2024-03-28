import functools
import numpy as np

N = 0
bitrate = []
quality = []

class Converter:
    def __init__(self, N_, bitrate_, quality_) -> None:
        global N, bitrate, quality
        N = N_
        bitrate = bitrate_
        quality = quality_
    
    @staticmethod
    @functools.lru_cache(maxsize=200)
    def n_to_b(n: int) -> int:
        return bitrate[n]
    
    @staticmethod
    @functools.lru_cache(maxsize=200)
    def b_to_n(b: int) -> int:
        return np.where(bitrate == b)[0][0]
    
    @staticmethod
    @functools.lru_cache(maxsize=200)
    def n_to_q(N_: int) -> int:
        return quality[N_]
    
    @staticmethod
    @functools.lru_cache(maxsize=200)
    def q_to_n(Q: int) -> int:
        return np.where(quality == Q)[0][0]
    
    @staticmethod
    @functools.lru_cache(maxsize=200)
    def b_to_q(b: int) -> int:
        return quality[np.where(bitrate == b)[0][0]]
    
    @staticmethod
    @functools.lru_cache(maxsize=200)
    def q_to_b(Q: int) -> int:
        return bitrate[np.where(quality == Q)[0][0]]
