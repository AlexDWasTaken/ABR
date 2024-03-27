import numpy as np

class Bandit:
    def __init__(self, N, alpha):
        self.__N = N
        self.kp_gain = np.zeros((N, N))
        self.sound_gain = np.zeros((N, N))
        self.__alpha = alpha

    def get_gains(self) -> np.ndarray:
        return self.kp_gain, self.sound_gain
    
    def calculate_initial_reward(self, raw_visual_quality, #Just for testing purpose
                         kp_func: function = lambda x: x+2,     
                         sound_func: function = lambda x: x+1,):
        sound_gain = np.vectorize(sound_func)(raw_visual_quality) - raw_visual_quality
        kp_gain = np.vectorize(kp_func)(raw_visual_quality) - raw_visual_quality
        return sound_gain, kp_gain

    def calculate_reward(self, raw_visual_quality: np.ndarray, 
                         I_sound: np.ndarray, I_video: np.ndarray, I_SR: np.ndarray, 
                         kp_cost: np.ndarray, sr_cost: np.ndarray, sound_cost: np.ndarray,
                         sr_func: function = lambda x: 2*x,     #Here, the functions are just for testing purposes
                         kp_func: function = lambda x: x+2,     #The actual functions should be passed as arguments
                         sound_func: function = lambda x: x+1,
                         ) -> tuple[np.ndarray, np.ndarray]:
        mask_sound = I_sound > 0.5
        mask_video = I_video > 0.5
        mask_SR = I_SR > 0.5

        sound_gain = np.zeros_like(raw_visual_quality)
        kp_gain = np.zeros_like(raw_visual_quality)
        sr_gain = np.zeros_like(raw_visual_quality)

        sound_gain[mask_sound] = (np.vectorize(sound_func)(raw_visual_quality) - raw_visual_quality)[mask_sound]
        kp_gain[mask_video] = (np.vectorize(kp_func)(raw_visual_quality) - raw_visual_quality)[mask_video]
        sr_gain[mask_SR] = (np.vectorize(sr_func)(raw_visual_quality) - raw_visual_quality)[mask_SR]

        #sound_ratio = sound_gain / (sound_cost + 1e-2) #avoid division by zero
        kp_ratio = kp_gain / (kp_cost + 1e-2)
        sr_ratio = sr_gain / (sr_cost + 1e-2)
        kp_gain_final = kp_gain * (kp_ratio > sr_ratio)

        return sound_gain, kp_gain_final
    
    def first_tick(self, raw_visual_quality,
                         kp_func: function = lambda x: x+2,     
                         sound_func: function = lambda x: x+1,):
        
        new_sound_gain, new_kp_gain = self.calculate_initial_reward(raw_visual_quality, kp_func, sound_func)
        self.kp_gain = new_kp_gain
        self.sound_gain = new_sound_gain

    def tick(self, raw_visual_quality, I_sound, I_video, I_SR, 
                         kp_cost: np.ndarray, sr_cost: np.ndarray, sound_cost: np.ndarray,
                         sr_func: function = lambda x: 2*x,     #Here, the functions are just for testing purposes
                         kp_func: function = lambda x: x+2,     #The actual functions should be passed as arguments
                         sound_func: function = lambda x: x+1,):
        new_sound_gain, new_kp_gain = self.calculate_reward(raw_visual_quality, I_sound, I_video, I_SR,
                                                    kp_cost, sr_cost, sound_cost, sr_func, kp_func, sound_func)
        self.kp_gain = (1 - self.__alpha) * self.kp_gain + self.__alpha * new_kp_gain
        self.sound_gain = (1 - self.__alpha) * self.sound_gain + self.__alpha * new_sound_gain
            
    def update_single(self, i, j, reward):
        self.__Q[i, j] = (1 - self.__alpha) * self.__Q[i, j] + self.__alpha * reward
    
    def update_all(self, rewards):
        self.__Q = (1 - self.__alpha) * self.__Q + self.__alpha * rewards