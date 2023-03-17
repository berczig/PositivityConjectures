"""
step 1 : generate uio
step 2: for any uio check all permutations and get the l,k correct sequences
step 3: From each l,k correct sequence extract this data:
    - last k+1 intervals
    - maximum interval from first l-1 intervals
    - p critical pairs from first intervals
    This data can be viewed as a graph and we classify all these graphs(from all uio) into the emerging categories.
step 4: e.g. linear programming

# maybe we don't have  to use ~ for the filter

"""

from itertools import permutations
from math import factorial as fac
import numpy as np
#import networkx as nx
import matplotlib.pyplot as plt 
import sys
import time 

INCOMPARABLE = 0
LE = 1
EQ = 2
GE = 3

permutations_n = {}
def getPermutationsOfN(n):
    # calculate all permutations of 1,..,n once - used multiple times
    global permutations_n
    if n not in permutations_n:
        permutations_n[n] = list(permutations(range(n)))
    return permutations_n[n]

def binomial(x, y):
    try:
        return fac(x) // fac(y) // fac(x - y)
    except ValueError:
        return 0

def C_n(n):
    return binomial(2*n, n)/(n+1)


def get_comparison_matrix(uio): # Relates to poset not incomparibility graph
    n = len(uio)
    C = np.diag([EQ for _ in range(n)]) # sets eq correct and all other to incomparable for now
    # 0: not comparable, 1 : less than, 2: equal 3: greater than
    for l in range(n):
        for k in range(l+1, n):
            if uio[k] > l:
                C[l, k] = LE
                C[k,l] = GE
    return C

def get_comparison_matrix_old(uio): # Relates to poset not incomparibility graph
    n = len(uio)
    C = np.zeros((n,n))
    #print("uio:", uio)
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
    print("Generated", len(A), "unit order intervals")
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
            if C[seq[i], seq[j]] in [LE, INCOMPARABLE]:
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
    return [seq for seq in getPermutationsOfN(n) if iscorrect(seq,C)]

def get_correct_ab_sequences(uio,a,b):
    n = len(uio)
    C = get_comparison_matrix(uio)
    return [seq for seq in getPermutationsOfN(n) if isabcorrect(seq,a,b,C)]

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
        if C[seq[i], seq[i-1]] == INCOMPARABLE:
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

def getsequencedatatomatrix(data, C):
    n = len(data)
    M = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            relation = C[data[i], data[j]]
            if relation == 0:
                relation = 2
            M[i,j] = relation
    return tuple(map(tuple, M))

def getsequencedatatovector(data, C):
    # vector of the edge-labels
    n = len(data)
    return tuple([C[data[i], data[j]] for i in range(n) for j in range(i+1, n)])

"""
def comparesequencedata(data1, data2, C):
    n = len(data1)
    for i in range(n):
        for j in range(i+1, n):
            edge1 = C[data1[i], data1[j]]
            edge2 = C[data2[i], data2[j]]
            if edge1 != edge2:
                return False
    return True"""

def getcoeff(uio, l, k):
    only_lk_correct = 0
    only_lplusk_correct = 0
    n = len(uio)
    C = get_comparison_matrix(uio)
    for seq in getPermutationsOfN(n):
        lk = isabcorrect(seq,l,k,C)
        lpk = iscorrect(seq, C)
        if lk and not lpk:
            only_lk_correct += 1
        elif lpk and not lk:
            only_lplusk_correct += 1
    #print("only_lplusk_correct:", only_lplusk_correct)
    return only_lk_correct - only_lplusk_correct

def evaluate_Conditions(lk_correct_seqs, Condition_matrix):
    # step 1: encode right condition into condition-MATRIX and check against all correct seqs
    # step2 : RL: deep neural cross entropy method
    # 
    # neural network policy structures:
    #   1: 45 or 46 , 60, 61, outputs for all the possible actions
    #   2: there is a 2. input telling you at what edge we are, just 4 outputs
    # 
    # Training happens like this:
    #   top 10 % become training data: conditions|->probability dist.
    #   train our current policy / neural network w.r.t. this classificaation. 
    pass

def getThml_2_coef(uio, l):
    n = len(uio)
    C = get_comparison_matrix(uio)
    A = []
    B = []
    for seq in getPermutationsOfN(n):
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
                if C[a,e] == GE and C[b,f] == GE:
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
    # [()]

def verifyl2Thm():
    # verify Theorem for l,2
    l = 6
    k = 2
    n = l+k
    print("verifyl2Thm")
    print("l:", l)
    print("k:", k)
    print("n:", n)
    A = generate_uio(n)
    for i,uio in enumerate(A):
        #uio_to_graph(uio)
        coef = getcoeff(uio,l,k)
        coef_thm = getThml_2_coef(uio, l)
        print(uio)
        if coef != coef_thm:
            print(i, "wrong", uio, coef, coef_thm)
        else:
            print(i, "right!", coef)    

def getCategories(uio_list, l, k):
    # uio_list : list of uios of same lengths
    #
    # lets say we have 2 datasequences:
    # (a1,b1,c1,d1,e1,f1)
    # (a2,b2,c2,d2,e2,f2)
    # and all edges have the same type but actually b1=c1 (so it has only 5 edges)
    # Should we put them in the same category? Yes, because w.r.t. the relation it's the same
    categories = {} # category:ID
    categories_counters = [] # list of dicts(category counters)
    n = len(uio_list[0])
    rows = len(uio_list)
    printval = 0
    for uio in uio_list:
        printval += 1
        if printval%10 == 0:
            pass
            #print(printval, "/",rows)
        counter = {}
        n = len(uio)
        C = get_comparison_matrix(uio)
        #print("C:", C)
        for seq in getPermutationsOfN(n):
            if isabcorrect(seq, l, k, C):
                # get sequence data
                data = getsequencedata(seq,C,p=1,l=l,k=k)
                #print("data:", data)
                M = getsequencedatatomatrix(data[1:], C)
                #print("M:", M)
                print("V:", getsequencedatatovector(data[1:], C))

                # determine category ID
                if M not in categories:
                    ID = len(categories)
                    categories[M] = ID
                else:
                    ID = categories[M]

                # count this observed category
                if ID not in counter:
                    counter[ID] = 1
                else:
                    counter[ID] += 1
        categories_counters.append(counter)

    # Turn collected category-count data into a matrix
    columns = len(categories)
    print("Found",columns, "categories")
    counts = []
    for i in range(rows):
        counter = categories_counters[i]
        row = [0 if j not in counter else counter[j] for j in range(columns)]
        counts.append(row)
    print("Computed count matrix of shape (", rows,",",columns, ")",sep="")
    return counts

if __name__ == "__main__":
    #print("yO", getThml_2_coef([0,0,0,0,0,0,1,2], 6))
    #verifyl2Thm()


    #Look at categories
    l = 4
    k = 2
    n = l+k
    print("l =",l)
    print("k =",k)
    print("n =",n)
    uios = generate_uio(n)
    categories = getCategories(uios, l, k)
    coeffs = np.array([getcoeff(uio,l,k) for uio in uios])

    print("First row of categories looks like:")
    print(categories[0])
    print()
    print("the first coefficient is:")
    print(coeffs[0])
    print("l, k correct sequences of all uio of length ", n, " is ", np.sum(np.array(categories)))

    # Solve Ax = b without linear programming, where A is count matrix and b the coefficients
    categories = np.array(categories)
    x = np.linalg.pinv(categories)@coeffs
    print("x:", [round(val, 3) for val in x])

    



    """
    for g in getcorrectsequences(uio):
        print("cor seq:", g)
    for g in get_correct_ab_sequences(uio, l, k):
        print("l 2 cor", g)"""

    """counting a,b correct sequences and step 3 data
    countcor = 0
    countabcor = 0
    for seq in getPermutationsOfN(n):
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