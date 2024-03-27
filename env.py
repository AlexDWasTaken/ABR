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
import datetime
from math import log2 as log

PACKET_SIZE = 1500.0
TIME_INTERVAL = 5.0
BITS_IN_BYTE = 8.0
MBITS_IN_BITS = 1000000.0
MILLISECONDS_IN_SECONDS = 1000.0

class Environment:
    def __init__(self, 
                 up_dataLoaders = [FCCDataLoader('202301_cooked_cleaned.txt', threshold=2000) for _ in range(3)], 
                 down_dataLoaders = [FCCDataLoader('202301_cooked_cleaned.txt',threshold=2000) for _ in range(3)],
                 frame_size: list[int] = [5, 4, 3, 2, 1, 0],
                 frame_q: list[float] = [log(5), log(4), log(3), log(2), log(1), -6],
                 client_weight: np.ndarray = np.array([4, 5, 1]), 
                 metrics: np.ndarray = np.array([11.4, -0.514, -0.1919, -0.810]), 
                 fps: float = 18, 
                 V: float = 1,
                 t0: float = 0.1,
                 alpha = 0.1,
                 kp_cost: np.ndarray = np.array([1, 1, 1]),
                 sr_cost: np.ndarray = np.array([1, 1, 1]),
                 sound_cost: np.ndarray = np.array([.5, .5, .5]),
                 sr_func: function = lambda x: x + 2 * np.log(np.abs(x)),     #Here, the functions are just for testing purposes
                 kp_func: function = lambda x: x+2,     #The actual functions should be passed as arguments
                 sound_func: function = lambda x: x+1,
                 msg: bool = True):
        
        self.up_dataloaders = up_dataLoaders
        self.down_dataloaders = down_dataLoaders
        self.N = len(up_dataLoaders)
        self.V = V
        self.t0 = t0
        self.frame_size = frame_size
        self.frame_q = frame_q
        self.client_weight = client_weight
        self.metrics = metrics
        self.fps = fps
        self.msg = msg
        self.kp_cost = kp_cost
        self.sr_cost = sr_cost
        self.sound_cost = sound_cost
        self.sr_func = sr_func
        self.kp_func = kp_func
        self.sound_func = sound_func
        self.q = np.zeros((self.N, self.N))
        self.d = np.zeros((self.N, self.N))
        self.QoE = []
        self.bandit = Bandit(self.N, alpha)
        self.iteration_count = 0
    
    def save_data(self, file_path):
        data = {
            'N': self.N,
            'q': self.q,
            'd': self.d,
            'QoE': self.QoE
        }
        np.savez(file_path, **data)

    def tick(self):
        # Update current network condition
        try:
            up_link, down_link = [], []
            for loader in self.up_dataloaders: 
                up_link.append(loader.tick())
            for loader in self.down_dataloaders: 
                down_link.append(loader.tick())
        except Exception as e:
            print(e)
            path = f'error{datetime.datetime.today()}.npz'
            print(f"An error occurred, saving available data to {path}")
            self.save_data(f'error{datetime.datetime.today()}.npz')
            return
        
        # Step 1: Do BaseLayerABR
        abr_parameters = {
            "N": self.N,
            "V": self.V,
            "t0": self.t0,
            "frame_size": self.frame_size,
            "frame_q": self.frame_q,
            "previous_q_ij": self.q,
            "previous_D": self.d,
            "up_link": np.array(up_link),
            "down_link": np.array(down_link),
            "client_weight": self.client_weight,
            "metrics": self.metrics,
            "fps": self.fps,
            "msg": True
        }
        abr = BaseLayerABR(**abr_parameters)
        abr.solve()
        send_option, receive_option = abr.get_options()
        send_bitrate = self.frame_size[send_option]
        receive_bitrate = self.frame_size[receive_option]
        raw_visual_quality = self.frame_q[receive_bitrate]


        # Step2: Estimate gains based on previous network conditions
        if self.iteration_count > 0:
            # Update the bandit.
            #TODO: Finish things up after getting I_kp and I_sr.
            bandit_parameters = {
                "raw_visual_quality": raw_visual_quality,
                "kp_func": self.kp_func,
                "sound_func": self.sound_func,
                "sr_func": self.sr_func
            }
        else:
            bandit_parameters = {
                "raw_visual_quality": raw_visual_quality,
                "kp_func": self.kp_func,
                "sound_func": self.sound_func
            }
            self.bandit.first_tick(**bandit_parameters)
            kp_gain, sound_gain = self.bandit.get_gains()
        
        # Step3: Use the estimated value and Semantic Selector to do the selection process.
        