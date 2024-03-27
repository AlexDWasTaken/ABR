import numpy as np
from pulp import LpMaximize, LpMinimize, LpProblem, LpStatus, lpSum, LpVariable, CPLEX_CMD, GUROBI_CMD, PULP_CBC_CMD, COIN_CMD

class BaseLayerABR:
    def __init__(self, N, V, t0, frame_size, previous_q_ij, previous_D, up_link, down_link, client_weight, metrics, fps, msg):
        self.problem = LpProblem(name="Base Layer ABR", sense=LpMinimize)
        self.msg = msg

        # Define Context
        self.N = N
        self.V = V
        self.t0 = t0
        self.frame_size = np.array(frame_size)
        self.frame_q = np.log(np.array(self.frame_size))
        self.choice_num = len(self.frame_size)
        self.previous_q_ij = previous_q_ij
        self.previous_D = previous_D

        self.s_min = self.frame_size[-1]
        self.s_max = self.frame_size[1]

        self.up_link = up_link
        self.down_link = down_link

        self.client_weight = np.abs(client_weight)
        self.metrics = np.array(metrics)
        self.fps = fps

        # Define Variables
        # One do not need to send to oneself, but the indicator will still be generated.
        self.send_option = [self.option_generator(self.problem, n=self.choice_num, name=f"send_option_{i}_") for i in range(self.N)]
        self.receive_option = [[self.option_generator(self.problem, n=self.choice_num, name=f"receive_option_{i}_to_{j}_") for j in range(self.N)] for i in range(self.N)]

        # Add constraints
        # Receive quality constraint
        for i in range(self.N):
            for j in range(self.N):
                for k in range(self.choice_num):
                    self.problem += (self.receive_option[i][j][k] <= lpSum((self.send_option[i][p] for p in range(0, k+1))))

        # Receive size constraint
        for j in range(self.N):
            total_receive_size = None
            for i in range(self.N):
                for k in range(self.choice_num):
                    if i == j: continue
                    if total_receive_size is None:
                        total_receive_size = self.receive_option[i][j][k] * self.frame_size[k]
                        continue
                    total_receive_size += self.receive_option[i][j][k] * self.frame_size[k]

            self.problem += (total_receive_size * self.fps <= self.down_link[j])

        # Send size constraint
        for i in range(self.N):
            total_send_size = None
            for k in range(self.choice_num):
                if total_send_size is None:
                    total_send_size = self.send_option[i][k] * self.frame_size[k]
                    continue
                total_send_size += self.send_option[i][k] * self.frame_size[k]
            self.problem += (total_send_size * self.fps <= self.up_link[i])


        # Define Objective
        # Step 1: QoE Related Objective
        # there are many 'if i!=j else 0.0' in the following functions, 
        # since one do not need to send to oneself and that part of QoE will be discarded.
        def get_send_size(i):
            return lpSum((self.frame_size[k] * self.send_option[i][k] for k in range(self.choice_num))) if i != j else 0.0

        def get_receive_size(i, j):
            return lpSum((self.frame_size[k] * self.receive_option[i][j][k] for k in range(self.choice_num))) if i != j else 0.0

        def get_q_ij(i, j):
            return lpSum((self.frame_q[k] * self.receive_option[i][j][k] for k in range(self.choice_num))) if i != j else 0.0

        def create_absolute_objective(problem, obj, var_name):
            sum = LpVariable(name=var_name, lowBound=0, cat="Continuous")
            problem += (sum >= obj)
            problem += (sum >= -obj)
            return sum

        def single_user_qoe(j):
            quality = lpSum((get_q_ij(i, j) for i in range(self.N)))
            variation = lpSum((create_absolute_objective(self.problem, get_q_ij(i, j) - self.previous_q_ij[i, j], f'abs_{i}_{j}') for i in range(self.N)))
            up_link_delay = get_send_size(j) / self.up_link[j]
            down_link_delay = lpSum((get_receive_size(i, j) / self.down_link[j] for i in range(self.N)))
            return self.metrics[0] * quality + self.metrics[1] * variation + self.metrics[2] * up_link_delay + self.metrics[3] * down_link_delay

        full_QoE = 0
        for j in range(self.N):
            full_QoE += -self.client_weight[j] * single_user_qoe(j)

        # Step2: Rate-Stability Related Objective
        stability = 0
        for i in range(self.N):
            for j in range(self.N):
                stability += self.previous_D[i, j] * (get_send_size(j) / self.up_link[j] + get_receive_size(i, j) / self.down_link[j] - self.t0)

        # Final Objective
        self.problem += self.V * full_QoE + stability

    def option_generator(self, problem: LpProblem, n, name):
        variables = [LpVariable(name=name+str(i), lowBound=0, upBound=1, cat="Binary") for i in range(n)]
        m = None
        for v in variables:
            m += v
        problem += (m == 1)
        return variables

    def solve(self, solver_msg=False):
        # Solve the problem
        solver = PULP_CBC_CMD(msg=solver_msg)
        status = self.problem.solve(solver=solver)
        print(f"status: {self.problem.status}, {LpStatus[self.problem.status]}") if self.msg else None
        return status, self.problem#.objective.value()
    
    def get_results(self):

        def convert_back(data):
            data_value = np.vectorize(lambda x: x.value())(data)
            data_converted = np.argmax(data_value, axis=-1)
            return data_converted
        
        send_option = convert_back(self.send_option)
        receive_option = convert_back(self.receive_option)
        #TODO: Return rest of the necessary values.

        return send_option, receive_option

    def __str__(self):
        return str(self.problem)


if __name__ == '__main__':

    test_parameters = {
        "N": 3,
        "V": 1,
        "t0": 0.1,
        "frame_size": [5, 4, 3, 2, 1],
        "previous_q_ij": np.zeros((3, 3)),
        "previous_D": np.zeros((3, 3)),
        "up_link": np.abs(np.random.rand(3)) * 1000,
        "down_link": np.abs(np.random.rand(3)) * 1000,
        "client_weight": [0.4, -0.2, -0.2, -0.1],
        "metrics": [11.4, -0.514, -0.1919, -0.810],
        "fps": 30,
        "msg": True
    }


    # Create an instance of the ABRProblem class
    abr = BaseLayerABR(**test_parameters)

    # Solve the problem
    abr.solve(solver_msg=True)