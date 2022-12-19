# step 1 : generate uio
# step 2: for any uio check all permutations and get the l,k correct sequences
# step 3: take last 3 digits + n critical pairs, look at type of that graph
# step 4: e.g. linear programming


from itertools import permutations
from math import factorial as fac
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt 
import sys

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

def isabcorrect(seq,a,b,C):
    return iscorrect(seq[:a], C) and iscorrect(seq[a:], C)

def iscorrect(seq, C):
    for i in range(1,len(seq)):
        # not to the left of previos interval
        if C[seq[i], seq[i-1]] == LE:
            return False
        # intersects with some previous interval
        intersects = False
        for j in range(0, i):
            if C[seq[i], seq[j]] in [LE, EQ]:
                intersects = True
                #break
        if not intersects:
            return False
    return True

def getcorrectsequences(uio, verbose=False):
    n = len(uio)
    C = get_comparison_matrix(uio)
    if verbose:
        print("uio:", uio)
        print("C", C)
    return [seq for seq in permutations(range(n)) if iscorrect(seq,C)]

def get_correct_ab_sequences(uio,a,b):
    n = len(uio)
    C = get_comparison_matrix(uio)
    return [seq for seq in permutations(range(n)) if isabcorrect(seq,a,b,C)]

def getmaximalinterval(subseq, C):
    maximals = []
    for i in subseq:
        ismaximal = True
        for j in subseq:
            if C[j, i] == GE: # one very redundant comparison: i==j, but whatever
                # i can't be maximal
                ismaximal = False
                break
        if ismaximal:
            maximals.append(i)
    return max(maximals)

def getsequencedata(seq,C, p, l, k): # step 3 for l,k
    data = []
    # take last critical pairs (at most p)
    n = l+k
    pairs = 0 # count number of registered critical pairs
    for i in range(l-1, 0, -1):
        if C[seq[i], seq[i-1]] == EQ:
            pairs += 1
            data.append(seq[i-1])
            data.append(seq[i])
            if pairs >= p:
                break
    data.insert(0, pairs)

    # maximal element in first l-1
    data.append(getmaximalinterval(seq[:l-1], C))

    # last k+1 elements
    data += seq[-(k+1):]
    return data

def getcoeff(uio, l, k):
    only_lk_correct = 0
    only_lplusk_correct = 0
    n = len(uio)
    C = get_comparison_matrix(uio)
    for seq in permutations(range(n)):
        lk = isabcorrect(seq,l,k,C)
        lpk = iscorrect(seq, C)
        if lk and not lpk:
            only_lk_correct += 1
        elif lpk and not lk:
            only_lplusk_correct += 1
    #print("only_lplusk_correct:", only_lplusk_correct)
    return only_lk_correct - only_lplusk_correct

def getThml_2_coef(uio, l):
    n = len(uio)
    C = get_comparison_matrix(uio)
    A = []
    B = []
    for seq in permutations(range(n)):
        if isabcorrect(seq, l, 2, C):
            data = getsequencedata(seq,C,p=1,l=l,k=2)
            #print("datax:", seq, data)
            if data[0] == 1:
                a,b,c,d,e,f = data[1:]
                allreadyinA = False
                if C[c,e] == LE and C[d,f] == LE:
                    A.append(data)
                    #print("A")
                    allreadyinA = True
                if C[e,a] == LE and C[f,b] == LE:
                    if allreadyinA:
                        print("A and B not disjoint!")
                        sys.exit()
                    B.append(data)
                    #print("B")
            else:
                print("Only found", data[0], "critical pairs in ",l,2,"!")
                sys.exit()
    #print("A:", len(A), "B:", len(B))
    return len(A) + len(B)

def verifyl2Thm():
    # verify Theorem for l,2
    l = 6
    k = 2
    n = l+k
    A = generate_uio(n)
    for i,uio in enumerate(A):
        #uio_to_graph(uio)
        coef = getcoeff(uio,l,k)
        coef_thm = getThml_2_coef(uio, l)
        if coef != coef_thm:
            print(i, "wrong", uio, coef, coef_thm)
        else:
            print(i, "right!", coef)    

"""
l = 6
k = 2
n = l+k
A = generate_uio(n)
uio = [0,0,0, 0,1,2,3,4]
print("There are", len(A), "uio of length", n)
C = get_comparison_matrix(uio)
for i,uio in enumerate(A):
    #uio_to_graph(uio)
    coef = getcoeff(uio,l,k)
    coef_thm = getThml_2_coef(uio, l)
    if coef != coef_thm:
        print(i, uio, coef, coef_thm)
    else:
        print("right!", coef)
"""

"""
for g in getcorrectsequences(uio):
    print("cor seq:", g)
for g in get_correct_ab_sequences(uio, l, k):
    print("l 2 cor", g)"""

"""counting a,b correct sequences and step 3 data
countcor = 0
countabcor = 0
for seq in permutations(range(n)):
    a = iscorrect(seq,C)
    b = isabcorrect(seq,l,k,C)
    if a:
        countcor += 1
    if b:
        countabcor += 1
    if a or b:
        print(seq, iscorrect(seq,C), b)
    print(getsequencedata(seq, C, 1, l, k))
print("countcor:", countcor)
print("countabcor:", countabcor)
"""

"""
K = 0
for uio in A:
    num = len(getcorrectsequences(uio))
    if num != 0:
        K += 1
        print("number of correct sequences in uio:", num)
print("uio with correct sequences:", K)
"""