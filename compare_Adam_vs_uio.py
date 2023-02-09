import Adam
import uio
import numpy as np
from itertools import permutations, combinations

print("Check my implementation vs Adam's")
l = 6
k = 2
n = l+k
print("l:", l)
print("k:", k)
print("n:", n)

# check if we generate the same uios
J =  uio.generate_uio(n)
A = Adam.catalan(n)
x = all([np.array_equal(A[i], J[i]) for i in range(len(A))])
print("Generate same uios?:", x)


all_permutations = list(permutations(list(range(n))))  # Generate all permutations of the vertices of the graph
all_permutations = [tuple(perm) for perm in all_permutations]  # Convert permutations to tuples

# check if we generate the same correct sequences given a uio
totalsame = True
for a_uio in A:
    Jcorseq = uio.getcorrectsequences(a_uio)
    Acorseq = Adam.find_all_correct_sequences(a_uio, all_permutations)
    same = sorted(Jcorseq) == sorted([tuple(corseq) for corseq in Acorseq])
    #print(len(Jcorseq), len(Acorseq))
    if same == False:
        print("Adam's and my generated correect sequences differ for the uio", a_uio)
        totalsame = False
print("Generating same correct sequences for all uio", totalsame)