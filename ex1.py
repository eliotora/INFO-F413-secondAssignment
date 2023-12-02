import numpy as np
from scipy.optimize import linprog

def primal(payoff_matrix):
    c = [-1] + [0 for _ in range(len(payoff_matrix))]

    Aub = [
        [1] + [-payoff_matrix[i][j] for i in range(len(payoff_matrix))] for j in range(len(payoff_matrix[0]))
    ]
    bub = [0 for _ in range(len(payoff_matrix[0]))]

    Aeq = [[0] + [1 for _ in range(len(payoff_matrix))]]
    beq = [1]

    bounds = [[None, None]]
    for _ in range(len(payoff_matrix)):
        bounds.append([0, None])

    res = linprog(c, Aub, bub, Aeq, beq, bounds, method="highs")

    return res.x[1:], -res.fun

def dual(payoff_matrix):
    c = [1] + [0 for _ in range(len(payoff_matrix[0]))]

    Aub = [
        [-1] + [payoff_matrix[i][j] for j in range(len(payoff_matrix[0]))] for i in range(len(payoff_matrix))
    ]
    bub = [0 for _ in range(len(payoff_matrix))]

    Aeq = [[0] + [1 for _ in range(len(payoff_matrix[0]))]]
    beq = [1]

    bounds = [[None, None]]
    for _ in range(len(payoff_matrix[0])):
        bounds.append([0, None])

    res = linprog(c, Aub, bub, Aeq, beq, bounds, method="highs")

    return res.x[1:], res.fun


if __name__ == "__main__":
    payoff_matrix = [
      [3, -1],
      [-2, 1]
    ]
    print(primal(payoff_matrix))
    print(dual(payoff_matrix))
