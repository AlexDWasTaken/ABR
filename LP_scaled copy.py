from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable, CPLEX_CMD, GUROBI_CMD, PULP_CBC_CMD
import random
import numpy as np
import numpy as np
import rich
from argparse import ArgumentParser

# python LP_scaled.py --save_to data.npz -n 80
# python LP_scaled.py --read_from data.npz

# 定义问题
N = 100

parser = ArgumentParser()
parser.add_argument("-n", type=int, default=100)
parser.add_argument("--read_from", type=str, default="-1")
parser.add_argument("--save_to", type=str, default="-1")
parser.add_argument("-s", type=bool, default=False)
parser.add_argument("-p", action="store_true")

args = parser.parse_args()
N = args.n

L = [random.uniform(-10, 0) for _ in range(N)]
R = [random.uniform(1, 10) for _ in range(N)]
G_kp = [[random.uniform(0, 10) for _ in range(N)] for _ in range(N)]
G_sound = [[random.uniform(0, 10) for _ in range(N)] for _ in range(N)]

def save_data_to_file(N, L, R, G_kp, G_sound, file_path):
    data = {
        'N': N,
        'L': L,
        'R': R,
        'G_kp': G_kp,
        'G_sound': G_sound
    }
    np.savez(file_path, **data)

def load_data_from_file(file_path):
    data = np.load(file_path)
    return data['N'], data['L'], data['R'], data['G_kp'], data['G_sound']

def save_data_to_txt(N, L, R, G_kp, G_sound, file_path):
    with open(file_path, "w") as f:
        f.write(f"{N}\n\n")
        f.write(" ".join([str(x) for x in L]) + "\n\n")
        f.write(" ".join([str(x) for x in R]) + "\n\n")
        for row in G_kp:
            f.write(" ".join([str(x) for x in row]) + "\n")
        f.write("\n")
        for row in G_sound:
            f.write(" ".join([str(x) for x in row]) + "\n")

def solve_with_sound(L, R, G_kp, G_sound, print_details=False, solver = PULP_CBC_CMD(msg=False)):
    model = LpProblem(name="not-big-problem", sense=LpMaximize)
    I_send = []
    I_receive_kp = [[] for _ in range(N)]
    I_receive_sound = [[] for _ in range(N)]

    for i in range(N):
        I_send.append(LpVariable(name=f"I_send_{i}", lowBound=0, upBound=1, cat="Integer"))

    for i in range(N):
        for j in range(N):
            I_receive_kp[i].append(LpVariable(name=f"I_receive_kp_{i}_{j}", lowBound=0, upBound=1, cat="Integer"))
            I_receive_sound[i].append(LpVariable(name=f"I_receive_sound_{i}_{j}", lowBound=0, upBound=1, cat="Integer"))

    alpha = 1
    beta = 1
    gamma = 1

    model += lpSum([I_send[i] * L[i] for i in range(N)]) + lpSum([I_receive_kp[i][j] * G_kp[i][j] for i in range(N) for j in range(N)]) + lpSum([I_receive_sound[i][j] * G_sound[i][j] for i in range(N) for j in range(N)])

    # 添加约束
    sum_kp, sum_sound, sum_send = None, None, None

    for j in range(N):
        for i in range(N):
            if i == j:
                continue
            if sum_kp is None:
                sum_kp = I_receive_kp[i][j]
                sum_sound = I_receive_sound[i][j]
            else:
                sum_kp += I_receive_kp[i][j]
                sum_sound += I_receive_sound[i][j]
        model += (sum_kp * alpha + sum_sound * beta + I_send[j] * gamma <= R[j])
        sum_kp, sum_sound, sum_send = None, None, None

    for i in range(N):
        for j in range(N):
            model += (I_send[i] >= I_receive_kp[i][j])

    # 求解问题
    status = model.solve(solver=solver)

    print(f"status: {model.status}, {LpStatus[model.status]}")
    print(f"objective: {model.objective.value()}")

    if print_details:
        for var in model.variables():
            print(f"model: {model}")
            print(f"{var.name}: {var.value()}")

    I_receive_kp_values = np.array([[I_receive_kp[i][j].value() for j in range(N)] for i in range(N)])
    I_receive_sound_values = np.array([[I_receive_sound[i][j].value() for j in range(N)] for i in range(N)])
    I_send_values = np.array([I_send[i].value() for i in range(N)])
    return model.objective.value(), I_receive_kp_values, I_receive_sound_values, I_send_values

def solve_without_sound(L, R, G_kp, G_sound, print_details=False, solver=PULP_CBC_CMD(msg=False)):
    model = LpProblem(name="not-big-problem", sense=LpMaximize)
    I_send = []
    I_receive_kp = [[] for _ in range(N)]
    #I_receive_sound = [[] for _ in range(N)]

    for i in range(N):
        I_send.append(LpVariable(name=f"I_send_{i}", lowBound=0, upBound=1, cat="Integer"))

    for i in range(N):
        for j in range(N):
            I_receive_kp[i].append(LpVariable(name=f"I_receive_kp_{i}_{j}", lowBound=0, upBound=1, cat="Integer"))
            #I_receive_sound[i].append(LpVariable(name=f"I_receive_sound_{i}_{j}", lowBound=0, upBound=1, cat="Integer"))

    alpha = 1
    beta = 1
    gamma = 1

    model += lpSum([I_send[i] * L[i] for i in range(N)]) + lpSum([I_receive_kp[i][j] * G_kp[i][j] for i in range(N) for j in range(N)])

    # 添加约束
    sum_kp, sum_sound, sum_send = None, 0, None

    for j in range(N):
        for i in range(N):
            if i == j:
                continue
            if sum_kp is None:
                sum_kp = I_receive_kp[i][j]
                #sum_sound = I_receive_sound[i][j]
            else:
                sum_kp += I_receive_kp[i][j]
                #sum_sound += I_receive_sound[i][j]
        model += (sum_kp * alpha + sum_sound * beta + I_send[j] * gamma <= R[j])
        sum_kp, sum_sound, sum_send = None, 0, None

    for i in range(N):
        for j in range(N):
            model += (I_send[i] >= I_receive_kp[i][j])

    # 求解问题
    status = model.solve(solver=solver)

    print(f"status: {model.status}, {LpStatus[model.status]}")
    print(f"objective: {model.objective.value()}")

    if print_details:
        for var in model.variables():
            print(f"model: {model}")
            print(f"{var.name}: {var.value()}")

    I_receive_kp_values = np.array([[I_receive_kp[i][j].value() for j in range(N)] for i in range(N)])
    #I_receive_sound_values = np.array([[I_receive_sound[i][j].value() for j in range(N)] for i in range(N)])
    I_send_values = np.array([I_send[i].value() for i in range(N)])
    return model.objective.value(), I_receive_kp_values, None, I_send_values


if args.read_from != "-1":
    N, L, R, G_kp, G_sound = load_data_from_file(args.read_from)
elif args.save_to != "-1":
    save_data_to_file(N, L, R, G_kp, G_sound, args.save_to)

if args.p:
    save_data_to_txt(N, L, R, G_kp, G_sound, "data.txt")

from timeit import default_timer

start = default_timer()
sound_result = solve_with_sound(L, R, G_kp, G_sound)
end = default_timer()
print(f"Time: {end - start}")

start = default_timer()
no_sound_result = solve_without_sound(L, R, G_kp, G_sound)
end = default_timer()
print(f"Time: {end - start}")

print(sound_result[0] - no_sound_result[0])
