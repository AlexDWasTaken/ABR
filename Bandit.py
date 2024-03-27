import numpy as np

class Bandit:
    #TODO: Implement Multiarmed Bandit class
    def __init__(self, N, alpha):
        self.__N = N
        self.__Q = np.zeros((N, N))
        self.__alpha = alpha
    
    def update_single(self, i, j, reward):
        self.__Q[i, j] = (1 - self.__alpha) * self.__Q[i, j] + self.__alpha * reward
    
    def update_all(self, rewards):
        self.__Q = (1 - self.__alpha) * self.__Q + self.__alpha * rewards

    def get_values(self):
        return self.__Q
    
    def calculate_reward(self, raw_bitrate, raw_visual_quality, I_sound, I_video, I_SR, 
                         kp_cost: np.ndarray, sr_cost: np.ndarray, sound_cost: np.ndarray,
                         sr_func: function = lambda x: 2*x, 
                         kp_func: function = lambda x: x+2, 
                         sound_func: function = lambda x: x+1,
                         ):
        mask_sound = I_sound > 0.5
        mask_video = I_video > 0.5
        mask_SR = I_SR > 0.5

        sound_gain = np.zeros_like(raw_bitrate)
        kp_gain = np.zeros_like(raw_bitrate)
        sr_gain = np.zeros_like(raw_visual_quality)

        sound_gain[mask_sound] = (np.vectorize(sound_func)(raw_bitrate) - raw_visual_quality)[mask_sound]
        kp_gain[mask_video] = (np.vectorize(kp_func)(raw_bitrate) - raw_visual_quality)[mask_video]
        sr_gain[mask_SR] = (np.vectorize(sr_func)(raw_visual_quality) - raw_visual_quality)[mask_SR]

        #sound_ratio = sound_gain / (sound_cost + 1e-2) #avoid division by zero
        kp_ratio = kp_gain / (kp_cost + 1e-2)
        sr_ratio = sr_gain / (sr_cost + 1e-2)


        return kp_gain * (kp_ratio > sr_ratio)