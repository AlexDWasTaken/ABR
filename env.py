"""
PACKET_SIZE = 1500.0  # bytes
TIME_INTERVAL = 5.0
BITS_IN_BYTE = 8.0
MBITS_IN_BITS = 1000000.0
MILLISECONDS_IN_SECONDS = 1000.0
N = 100
LINK_FILE = './cooked/trace_390_http---m.imdb.com-help-'


bandwidth_all = []
with open(LINK_FILE, 'rb') as f:
	for line in f:
		throughput = int(line.split()[0])
		bandwidth_all.append(throughput)

bandwidth_all = np.array(bandwidth_all)
bandwidth_all = bandwidth_all * BITS_IN_BYTE / MBITS_IN_BITS
"""

from BaseLayerABR import BaseLayerABR
from SemanticSelector import SemanticSelector
from Bandit import Bandit
from datasets.fcc.FCCDataLoader import FCCDataLoader, FCCRecoveryDataLoader
import numpy as np

PACKET_SIZE = 1500.0
TIME_INTERVAL = 5.0
BITS_IN_BYTE = 8.0
MBITS_IN_BITS = 1000000.0
MILLISECONDS_IN_SECONDS = 1000.0

class Environment:
    def __init__(self, dataLoaders: list[FCCDataLoader], frame_size: list[int], client_weight: np.ndarray, metrics: np.ndarray, fps: int, msg: bool):
        self.dataloaders = dataLoaders
        self.N = len(dataLoaders)
        self.q = np.zeros((self.N, self.N))
        self.d = np.zeros((self.N, self.N))