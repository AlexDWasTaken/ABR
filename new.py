import numpy as np
from pulp import LpMaximize, LpMinimize, LpProblem, LpStatus, lpSum, LpVariable, CPLEX_CMD, GUROBI_CMD, PULP_CBC_CMD

problem = LpProblem(name="Base Layer ABR", sense=LpMinimize)
msg = True

# Define Context
N = num_clients = 30
V = 0.8
t0 = 0.5
frame_size = [5, 4, 3, 2, 1]
frame_q = np.log(np.array(frame_size))
choice_num = len(frame_size)
previous_q_ij = np.random.rand(N, N)
previous_D = 1/10+1/5*np.random.rand(N, N)

s_min = frame_size[-1]
s_max = frame_size[1]

up_link = np.abs(np.random.rand(N)) * 1000
down_link = np.abs(np.random.rand(N)) * 1000

client_weight = np.abs(np.random.rand(N)) * 3
metrics = np.array([11.4, -0.514, -0.1919, -0.810])
fps = 20

# Define Variables
def option_generator(problem: LpProblem, n, name):
    variables = [LpVariable(name=name+str(i), lowBound=0, upBound=1, cat="Binary") for i in range(n)]
    m = None
    for v in variables:
        m += v
    problem += (m == 1)
    return variables

send_option = [option_generator(problem, n=N, name=f"send_option_{i}_") for i in range(N)]
receive_option = [[option_generator(problem, n=N, name=f"receive_option_{i}_to_{j}_") for j in range(N)] for i in range(N)]

# Add constraints
# Receive quality constraint
for i in range(N):
    for j in range(N):
        for k in range(choice_num):
            problem += (receive_option[i][j][k] <= lpSum((send_option[i][p] for p in range(0, k+1))))


# Receive size constraint
for j in range(N):
    total_receive_size = None
    for i in range(N):
        for k in range(choice_num):
            if i == j: continue
            if total_receive_size is None:
                total_receive_size = receive_option[i][j][k] * frame_size[k]
                continue
            total_receive_size += receive_option[i][j][k] * frame_size[k]

    problem += (total_receive_size * fps <= down_link[j])

# Send size constraint
for i in range(N):
    total_send_size = None
    for k in range(choice_num):
        if total_send_size is None:
            total_send_size = send_option[i][k] * frame_size[k]
            continue
        total_send_size += send_option[i][k] * frame_size[k]
    problem += (total_send_size * fps <= up_link[i])

# Define Objective
# Step 1: QoE Related Objective
def get_send_size(i):
    return lpSum((frame_size[k] * send_option[i][k] for k in range(choice_num))) if i != j else 0.0

def get_receive_size(i, j):
    return lpSum((frame_size[k] * receive_option[i][j][k] for k in range(choice_num))) if i != j else 0.0

def get_q_ij(i, j):
    return lpSum((frame_q[k] * receive_option[i][j][k] for k in range(choice_num))) if i != j else 0.0

def create_absolute_objective(problem, obj, var_name):
    sum = LpVariable(name=var_name, lowBound=0, cat="Continuous")
    problem += (sum >= obj)
    problem += (sum >= -obj)
    return sum

def single_user_qoe(j):
    quality = lpSum((get_q_ij(i, j) for i in range(N)))
    variation = lpSum(
        (create_absolute_objective(problem, get_q_ij(i, j) - previous_q_ij[i, j], f'abs_{i}_{j}') for i in range(N))
        )
    up_link_delay = get_send_size(j) / up_link[j]
    down_link_delay = lpSum((get_receive_size(i, j) / down_link[j] for i in range(N)))
    return metrics[0] * quality + metrics[1] * variation + metrics[2] * up_link_delay + metrics[3] * down_link_delay

full_QoE = 0
for j in range(N):
    full_QoE += -client_weight[j] * single_user_qoe(j)

# Step2: Rate-Stability Related Objective
stability = 0
for i in range(N):
    for j in range(N):
        stability += previous_D[i, j] * (get_send_size(j) / up_link[j] + get_receive_size(i, j) / down_link[j] - t0)

#Final Objective
print(type(full_QoE))
problem += V * full_QoE + stability

print(problem)

# Solve the problem
solver = PULP_CBC_CMD(msg=msg)
status = problem.solve(solver=solver)
print(f"status: {problem.status}, {LpStatus[problem.status]}")