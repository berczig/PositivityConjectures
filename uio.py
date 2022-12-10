# step 1 : generate uio
# step 2: for any uio check all permutations all get he correct sequences
# step 3: take last 3 digits + n critical pairs, look at type of that graph

from itertools import permutations
from math import factorial as fac


def binomial(x, y):
    try:
        return fac(x) // fac(y) // fac(x - y)
    except ValueError:
        return 0

def C_n(n):
    return binomial(2*n, n)/(n+1)

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

def getcorrectsequences(uio):
    print("uio:", uio)
    for seq in permutations(uio):
        print(seq)

A = generate_uio(4)
getcorrectsequences(A[7])