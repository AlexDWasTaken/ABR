import numpy as np

def adapt_strategy(N, I_reveive_kp: np.ndarray, I_receive_sound: np.ndarray, I_send_kp: np.ndarray, 
                   up_link: list[int], down_link: list[int], receive_option: np.ndarray,
                   send_bitrate: np.ndarray, receive_bitrate: np.ndarray, 
                   cost_constraints: np.ndarray, frame_size: list[int], msg=False):
    I_reveive_kp, I_receive_sound, I_send_kp = I_reveive_kp.copy(), I_receive_sound.copy(), I_send_kp.copy()
    total_options = len(frame_size)

    #Firstly, do bandwidth decisions.
    for user_index in range(N):
        user_uplink, user_downlink = up_link[user_index], down_link[user_index]
        user_send_bitrate, user_receive_bitrate = send_bitrate[user_index], receive_bitrate[:, user_index]
        user_receive_option = receive_option[:, user_index]
        user_receive_sound, user_receive_kp = I_receive_sound[:, user_index], I_reveive_kp[user_index]

        if msg:
            print("="*20)
            print("User", user_index)


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

        
        

    result = {
            "receive_option": receive_option,
            "receive_bitrate": frame_size[receive_option],
            "I_receive_sr": None
    }
    return result



if __name__ == "__main__":
    N = 3
    frame_size = np.array([5, 4, 3, 2, 1])
    I_reveive_kp = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    I_receive_sound = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    I_send_kp = np.array([1, 1, 1])
    up_link = [8, 10, 9]
    down_link = [7, 5, 8]
    receive_option = np.array([[4, 4, 4], [4, 4, 4], [4, 4, 4]])
    send_bitrate = np.array([1, 1, 1])
    receive_bitrate = np.array(frame_size[receive_option])
    cost_constraints = np.array([1, 1, 1])

    result = adapt_strategy(N, I_reveive_kp, I_receive_sound, I_send_kp, up_link, down_link, receive_option, send_bitrate, receive_bitrate, cost_constraints, frame_size, True)
    import rich
    for key, value in result.items():
        rich.print(f"{key}:")
        rich.print(value)