from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable, CPLEX_CMD, GUROBI_CMD, PULP_CBC_CMD
import random
import numpy as np
import numpy as np
import rich
from argparse import ArgumentParser

class SemanticSelector:
    def __init__(self, N, L, R, G_kp, G_sound, msg=False):
        self.N = N
        self.L = L
        self.R = R
        self.G_kp = G_kp
        self.G_sound = G_sound
        self.msg = msg

    def save_data_to_file(self, file_path):
        data = {
            'N': self.N,
            'L': self.L,
            'R': self.R,
            'G_kp': self.G_kp,
            'G_sound': self.G_sound
        }
        np.savez(file_path, **data)

    @staticmethod
    def load_data_from_file(file_path):
        data = np.load(file_path)
        return data['N'], data['L'], data['R'], data['G_kp'], data['G_sound']

    @staticmethod
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

    def solve_with_sound(self, print_details=False, solver=PULP_CBC_CMD):
        solver = solver(msg=self.msg)
        model = LpProblem(name="not-big-problem", sense=LpMaximize)
        I_send = []
        I_receive_kp = [[] for _ in range(self.N)]
        I_receive_sound = [[] for _ in range(self.N)]

        for i in range(self.N):
            I_send.append(LpVariable(name=f"I_send_{i}", lowBound=0, upBound=1, cat="Integer"))

        for i in range(self.N):
            for j in range(self.N):
                I_receive_kp[i].append(LpVariable(name=f"I_receive_kp_{i}_{j}", lowBound=0, upBound=1, cat="Integer"))
                I_receive_sound[i].append(LpVariable(name=f"I_receive_sound_{i}_{j}", lowBound=0, upBound=1, cat="Integer"))

        alpha = 1
        beta = 1
        gamma = 1

        model += lpSum([I_send[i] * self.L[i] for i in range(self.N)]) + lpSum([I_receive_kp[i][j] * self.G_kp[i][j] for i in range(self.N) for j in range(self.N)]) + lpSum([I_receive_sound[i][j] * self.G_sound[i][j] for i in range(self.N) for j in range(self.N)])

        # 添加约束
        sum_kp, sum_sound, sum_send = None, None, None

        for j in range(self.N):
            for i in range(self.N):
                if i == j:
                    continue
                if sum_kp is None:
                    sum_kp = I_receive_kp[i][j]
                    sum_sound = I_receive_sound[i][j]
                else:
                    sum_kp += I_receive_kp[i][j]
                    sum_sound += I_receive_sound[i][j]
            model += (sum_kp * alpha + sum_sound * beta + I_send[j] * gamma <= self.R[j])
            sum_kp, sum_sound, sum_send = None, None, None

        for i in range(self.N):
            for j in range(self.N):
                model += (I_send[i] >= I_receive_kp[i][j])

        # 求解问题
        status = model.solve(solver=solver)

        print(f"status: {model.status}, {LpStatus[model.status]}")
        print(f"objective: {model.objective.value()}")

        if print_details:
            for var in model.variables():
                print(f"model: {model}")
                print(f"{var.name}: {var.value()}")

        I_receive_kp_values = np.array([[I_receive_kp[i][j].value() for j in range(self.N)] for i in range(self.N)])
        I_receive_sound_values = np.array([[I_receive_sound[i][j].value() for j in range(self.N)] for i in range(self.N)])
        I_send_values = np.array([I_send[i].value() for i in range(self.N)])
        return model.objective.value(), I_receive_kp_values, I_receive_sound_values, I_send_values

    def solve_without_sound(self, print_details=False, solver=PULP_CBC_CMD):
        solver = solver(msg=self.msg)
        model = LpProblem(name="not-big-problem", sense=LpMaximize)
        I_send = []
        I_receive_kp = [[] for _ in range(self.N)]
        #I_receive_sound = [[] for _ in range(self.N)]

        for i in range(self.N):
            I_send.append(LpVariable(name=f"I_send_{i}", lowBound=0, upBound=1, cat="Integer"))

        for i in range(self.N):
            for j in range(self.N):
                I_receive_kp[i].append(LpVariable(name=f"I_receive_kp_{i}_{j}", lowBound=0, upBound=1, cat="Integer"))
                #I_receive_sound[i].append(LpVariable(name=f"I_receive_sound_{i}_{j}", lowBound=0, upBound=1, cat="Integer"))

        alpha = 1
        beta = 1
        gamma = 1

        model += lpSum([I_send[i] * self.L[i] for i in range(self.N)]) + lpSum([I_receive_kp[i][j] * self.G_kp[i][j] for i in range(self.N) for j in range(self.N)])

        # 添加约束
        sum_kp, sum_sound, sum_send = None, 0, None

        for j in range(self.N):
            for i in range(self.N):
                if i == j:
                    continue
                if sum_kp is None:
                    sum_kp = I_receive_kp[i][j]
                    #sum_sound = I_receive_sound[i][j]
                else:
                    sum_kp += I_receive_kp[i][j]
                    #sum_sound += I_receive_sound[i][j]
            model += (sum_kp * alpha + sum_sound * beta + I_send[j] * gamma <= self.R[j])
            sum_kp, sum_sound, sum_send = None, 0, None

        for i in range(self.N):
            for j in range(self.N):
                model += (I_send[i] >= I_receive_kp[i][j])

        # 求解问题
        status = model.solve(solver=solver)

        print(f"status: {model.status}, {LpStatus[model.status]}")
        print(f"objective: {model.objective.value()}")

        if print_details:
            for var in model.variables():
                print(f"model: {model}")
                print(f"{var.name}: {var.value()}")

        I_receive_kp_values = np.array([[I_receive_kp[i][j].value() for j in range(self.N)] for i in range(self.N)])
        #I_receive_sound_values = np.array([[I_receive_sound[i][j].value() for j in range(self.N)] for i in range(self.N)])
        I_send_values = np.array([I_send[i].value() for i in range(self.N)])
        return model.objective.value(), I_receive_kp_values, None, I_send_values
    

if __name__ == '__main__':
    N = 10
    parser = ArgumentParser()
    parser.add_argument("-n", type=int, default=10)
    parser.add_argument("--read_from", type=str, default="-1")
    parser.add_argument("--save_to", type=str, default="-1")
    parser.add_argument("-s", type=bool, default=False)
    parser.add_argument("-msg", action="store_true")

    args = parser.parse_args()
    N = args.n

    L = [random.uniform(-10, 0) for _ in range(N)]
    R = [random.uniform(1, 10) for _ in range(N)]
    G_kp = [[random.uniform(0, 10) for _ in range(N)] for _ in range(N)]
    G_sound = [[random.uniform(0, 10) for _ in range(N)] for _ in range(N)]


    selector = SemanticSelector(N, L, R, G_kp, G_sound, msg=args.msg)

    print(selector.solve_with_sound(print_details=False))
    print(selector.solve_without_sound(print_details=False))