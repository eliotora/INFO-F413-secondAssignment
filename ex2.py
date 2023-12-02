from itertools import product
import numpy as np
from tqdm import tqdm
from ex1 import primal, dual
import matplotlib.pyplot as plt

class Leaf:
    def __init__(self, val, index):
        self.looked = False
        self.val = val
        self.index = index

    def search(self):
        self.looked = True
        return self.val

    def print(self):
        print(self.val, end="")

    def reset(self):
        self.looked = False

    def new_val(self, tree_bin, seq_bin):
        self.reset()
        self.val = seq_bin.pop(0)

class Node:
    def __init__(self, left_firt):
        self.bit = left_firt
        self.left_first = left_firt == 1
        self.L = None
        self.R = None
        self.looked = False

    def search(self):
        self.looked = True
        if self.left_first:
            if self.L.search() == 1:
                return 0
            elif self.R.search() == 1:
                return 0
            else:
                return 1
        else:
            if self.R.search() == 1:
                return 0
            elif self.L.search() == 1:
                return 0
            else:
                return 1

    def print(self):
        self.L.print()
        self.R.print()

    def new_val(self, tree_bin, seq_bin):
        self.looked = False
        self.left_first = tree_bin.pop(0)
        self.L.new_val(tree_bin, seq_bin)
        self.R.new_val(tree_bin, seq_bin)


def create_tree(k):
    leafs = [Leaf(0, i) for i in range(4**k)]
    tree = leafs.copy()

    while len(tree) > 1:
        n = Node(0)
        n.L = tree.pop(0)
        n.R = tree.pop(0)
        tree.append(n)

    return tree[0], leafs

def random_sequences(length, number):
    ret = []
    for i in range(number):
        seq = "".join([str(c) for c in np.random.randint(0, 2, length)])
        while seq in ret:
            seq = "".join([str(c) for c in np.random.randint(0, 2, length)])
        ret.append(seq)
    return ret

def get_payoff_matrix(k):
    # min_a = 2**(2**k) - 1
    min_a = min(2**(2**k) - 1, 4**k)
    payoffs = np.array([[None for _ in range(min_a)] for _ in range(4 ** k)])
    # payoffs = np.array([[None for _ in range(2**(2**(2**k)-1))] for _ in range(2**(4 ** k))])
    tree, leafs = create_tree(k)

    # Generate 4**k random sequences with no duplicates for I
    I = random_sequences(4 ** k, 4 ** k)
    # I = random_sequences(4 ** k, 2**(4 ** k))

    # Generate 2**(2**k) - 1 random sequences with no duplicates for A
    A = random_sequences(2 ** (2 ** k) - 1, min_a)
    # A = random_sequences(2 ** (2 ** k) - 1, 2**(2**(2**k)-1))

    for i in tqdm(range(len(payoffs))):
        In = I[i]
        for j in range(len(payoffs[0])):
            Alg = A[j]
            Icopy = [int(c) for c in In]
            Algcopy = [int(c) for c in Alg]
            tree.new_val(Algcopy, Icopy)
            tree.search()
            payoffs[i, j] = sum([1 if l.looked else 0 for l in leafs])

    return I, A, payoffs


if __name__ == "__main__":
    k = 4
    p_payoffs, d_payoffs = [], []
    p_solutions, d_solutions = [], []
    nb_iter = 10
    i_s = [0 for _ in range(4**k)]
    a_s = [0 for _ in range(2**(2**k))]

    for i in tqdm(range(nb_iter)):
        I, A, pm = get_payoff_matrix(k)
        # print(pm)
        p_solution, p_expected_payoff = primal(pm)

        for i in range(len(I)):
            # print(I[i])
            # print(4**k//(2*k))
            # print(I[i][::-4**k//(2*k)])
            i_s[int(I[i][::-4**k//(2*k)], 2)] += p_solution[i]

        # print(p_solution, p_expected_payoff)

        p_payoffs.append(p_expected_payoff)
        p_solutions.append(np.array(p_solution))

        d_solution, d_expected_payoff = dual(pm)
        # print(d_solution, d_expected_payoff)

        for a in range(len(A)):
            a_s[int(A[a][::-2**(2**k)//(2**k)], 2)] += d_solution[a]

        d_payoffs.append(d_expected_payoff)
        d_solutions.append(np.array(d_solution))


    print(sum(p_payoffs)/nb_iter)
    print(sum(d_payoffs)/nb_iter)
    print(sum(p_solutions)/nb_iter)
    print(sum(d_solutions)/nb_iter)
    best_in = np.array(i_s)/nb_iter
    print(best_in)
    best_tree = np.array(a_s)/nb_iter
    print(best_tree)

    plt.plot(np.arange(len(best_tree)), best_tree, "r-")
    # plt.ylim(0, 10/2**(2**k))
    plt.title("Best algorithm (column player), bucketed on "+ str(2**k) + " bits")
    plt.xlabel("Algorithm bucket")
    plt.ylabel("Probability distribution")
    plt.show()
    plt.plot(np.arange(len(best_in)), best_in, "b-")
    # plt.ylim(0, 10/2**(2**k))
    plt.title("Best input (row player), bucketed on "+ str(2**k) + " bits")
    plt.xlabel("Input bucket")
    plt.ylabel("Probability distribution")
    plt.show()
