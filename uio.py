# step 1 : generate uio
# step 2: for any uio check all permutations all get he correct sequences
# step 3: take last 3 digits + n critical pairs, look at type of that graph

from itertools import permutations
from math import factorial as fac
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt 

LE = 1
EQ = 2
GE = 3

def binomial(x, y):
    try:
        return fac(x) // fac(y) // fac(x - y)
    except ValueError:
        return 0

def C_n(n):
    return binomial(2*n, n)/(n+1)

def get_comparison_matrix(uio):
    n = len(uio)
    C = np.zeros((n,n))
    # 0: not comparable, 1 : less than, 2: equal 3: greater than
    for i in range(n):
        for j in range(i+1, n):
            if uio[j] <= i:
                C[i,j] = EQ
                C[j,i] = EQ
            else:
                C[i, j] = LE
                C[j,i] = GE
    return C

def uio_to_graph(uio):
    G=nx.Graph()
    n = len(uio)
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i+1, n):
            if uio[j] <= i:
                G.add_edge(i,j)
    nx.draw(G)
    plt.show()

def generate_uio(n):
    A = []
    generate_uio_rec(A, [0], n, 1)
    return A

def generate_uio_rec(A, uio, n, i):
    if i == n:
        A.append(uio)
        return
    for j in range(uio[i-1], i+1):
        generate_uio_rec(A, uio+[j], n, i+1)

def iscorrect(seq, C):
    for i in range(1,len(seq)):
        # not to the left of previos interval
        if C[seq[i], seq[i-1]] == LE:
            return False
        # intersects with some previous interval
        intersects = False
        for j in range(0, i):
            if C[seq[i], seq[j]] != GE:
                intersects = True
                #break
        if not intersects:
            return False
    return True

def getcorrectsequences(uio):
    n = len(uio)
    C = get_comparison_matrix(uio)
    print("uio:", uio)
    print("C", C)
    for seq in permutations(range(n)):
        print("seq:", [i+1 for i in seq], iscorrect(seq,C))

A = generate_uio(4)
#get_comparison_matrix([0,0,0,1,2])
getcorrectsequences([0,0,1,2,3])
