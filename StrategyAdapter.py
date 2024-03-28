import numpy as np
from QualityConverter import Converter

def adapt_strategy(N, I_reveive_kp: np.ndarray, I_receive_sound: np.ndarray, I_send_kp: np.ndarray, 
                   up_link: list[int], down_link: list[int], receive_option: np.ndarray,
                   send_bitrate: np.ndarray, receive_bitrate: np.ndarray, raw_receive_QoE: np.ndarray,
                   cost_constraints: np.ndarray, frame_size: list[int], converter: Converter, 
                   kp_cost: np.ndarray, sr_cost: np.ndarray, kp_compute_cost: np.ndarray, sound_cost: np.ndarray,
                    sr_gain, msg=False) -> dict:
    I_reveive_kp, I_receive_sound, I_send_kp = I_reveive_kp.copy(), I_receive_sound.copy(), I_send_kp.copy()
    total_options = len(frame_size)

    I_receive_sr = np.zeros((N, N))
    for user_index in range(N):
        user_uplink, user_downlink = up_link[user_index], down_link[user_index]
        user_send_bitrate, user_receive_bitrate = send_bitrate[user_index], receive_bitrate[:, user_index]
        user_receive_option = receive_option[:, user_index]
        user_receive_sound, user_receive_kp = I_receive_sound[:, user_index], I_reveive_kp[user_index]
        user_I_receive_sr = I_receive_sr[:, user_index]
        user_sr_cost = sr_cost[user_index]
        user_kp_cost = kp_cost[user_index]
        user_kp_compute_cost = kp_compute_cost[user_index]
        user_sound_cost = sound_cost[user_index]
        user_raw_QoE = raw_receive_QoE[:, user_index]

        if msg:
            print("=*"*20)
            print("User", user_index)

        #Firstly, do bandwidth decisions.
        # Free the bandwidth for the user
        free_bandwidth = user_downlink - np.sum(np.logical_or(user_receive_sound, user_receive_kp) * user_receive_bitrate)
        #sorted_indices = np.argsort(user_receive_bitrate) # Sort the indices of the receive_bitrate from low to high

        # Fill the free bandwidth
        if msg:
            print("free bandwidth:", free_bandwidth) if msg else None
            print("user_receive_bitrate:", user_receive_bitrate) if msg else None

        for i in range(total_options-1):
            option_to_enhance = total_options - i - 1
            indices = np.where(user_receive_option == option_to_enhance)[0]
            if msg: 
                print(f"Enhancing option {option_to_enhance}, deciding whether to upgrade {frame_size[option_to_enhance]} to {frame_size[option_to_enhance-1]}")
                print("user_receive_options:", user_receive_option)
                print("indices to look at:", indices)
                print("free_bandwidth:", free_bandwidth)

            if free_bandwidth > 0:
                bandwidth_to_use = frame_size[option_to_enhance-1] - frame_size[option_to_enhance]
                if msg: print(f"Spending {bandwidth_to_use} to upgrade for each user.")
                for index in indices:
                    if msg: 
                        print(f"User {index} is upgrading.")
                    
                    if free_bandwidth >= bandwidth_to_use:
                        user_receive_option[index] -= 1
                        free_bandwidth -= bandwidth_to_use
                    else:
                        break
            else:
                break
        # Secondly, finalize sr decisions.
        free_resource = cost_constraints[user_index] - np.sum(user_receive_kp) * user_kp_cost - np.sum(user_receive_sound) * user_sound_cost - I_send_kp[user_index] * user_kp_compute_cost
        
        if msg:
            print("--"*10)
            print("Finalizing SR decisions.")
            print(free_resource)
        while free_resource >= user_sr_cost:
            # Find the best option to upgrade
            things = np.vectorize(sr_gain)(user_receive_bitrate) - np.vectorize(converter.b_to_q)(user_receive_bitrate) - 100 * user_I_receive_sr
            best_option = np.argmax(things)
            if msg:
                print("best_option:", best_option)

            user_I_receive_sr[best_option] = 1
            free_resource -= user_sr_cost


    result = {
            "receive_option": receive_option,
            "receive_bitrate": frame_size[receive_option],
            "I_receive_sr": I_receive_sr
    }
    return result



if __name__ == "__main__":
    N = 3
    frame_size = np.array([5, 4, 3, 2, 1])
    I_reveive_kp = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0]])
    I_receive_sound = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    I_send_kp = np.array([1, 1, 1])
    up_link = [8, 10, 9]
    down_link = [7, 5, 8]
    receive_option = np.array([[4, 4, 4], [4, 4, 4], [4, 4, 4]])
    send_bitrate = np.array([1, 1, 1])
    quality = np.log2(frame_size)
    receive_bitrate = np.array(frame_size[receive_option])
    raw_receive_QoE = np.log(receive_bitrate) + 10
    cost_constraints = np.array([4, 4, 3])
    kp_cost = np.array([1] * 3)
    sr_cost = np.array([0.8] * 3)
    sound_cost = np.array([0.5] * 3)
    kp_compute_cost = np.array([0.5] * 3)
    converter = Converter(N, frame_size, quality)
    sr_gain = lambda x: x + 2 * np.log(np.abs(x))

    test_arguments = {
        "N": N,
        "I_reveive_kp": I_reveive_kp,
        "I_receive_sound": I_receive_sound,
        "I_send_kp": I_send_kp,
        "up_link": up_link,
        "down_link": down_link,
        "receive_option": receive_option,
        "send_bitrate": send_bitrate,
        "receive_bitrate": receive_bitrate,
        "cost_constraints": cost_constraints,
        "frame_size": frame_size,
        "msg": True,
        "kp_cost": kp_cost,
        "sr_cost": sr_cost,
        "sound_cost": sound_cost,
        "kp_compute_cost": kp_compute_cost,
        "converter": converter,
        "sr_gain": sr_gain,
        "raw_receive_QoE": raw_receive_QoE
    }

    result = adapt_strategy(**test_arguments)
    import rich
    for key, value in result.items():
        rich.print(f"{key}:")
        rich.print(value)