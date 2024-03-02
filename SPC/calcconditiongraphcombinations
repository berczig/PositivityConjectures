from math import factorial as fac
from itertools import product
def binomial(x, y):
    try:
        return fac(x) // fac(y) // fac(x - y)
    except ValueError:
        return 0
def biggraph(n,k,e,L):
    edges = binomial(n,2)
    pickededges = binomial(edges, e)
    return pickededges*(L*k)**e
def multiplegraph(n,k,e,L):
    A = list(range(e+1))
    cartesian = product(A,A,A)
    equalpartitions = []
    for k in cartesian:
        if sum(k) == e:
            equalpartitions.append(k)
    SUM = 0
    for S in equalpartitions:
        counts = 1
        for e in S:
            counts *= biggraph(n,1,e,L)
        SUM += counts
    return SUM
def compare(n,k,e,L):
    print(biggraph(n,k,e,L))
    print(multiplegraph(n,k,e,L))

compare(6,2,4,3)
compare(9,3,9,3)
#print(biggraph(6,2,4,3))
#print(biggraph(9,2,6,3))
#print(biggraph(9,3,9,3))
#print(multiplegraph(6,2,4,3))
