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
# check if we get the same l,2 coef given a uio
totalsame = True
totalsamelk = True
for a_uio in A:
    Jcorseq = uio.getcorrectsequences(a_uio)
    Acorseq = Adam.find_all_correct_sequences(a_uio, all_permutations)
    same = sorted(Jcorseq) == sorted([tuple(corseq) for corseq in Acorseq])
    #print(len(Jcorseq), len(Acorseq))
    if same == False:
        print("Adam's and my generated correct sequences differ for the uio", a_uio)
        totalsame = False

    Alkcorseq = Adam.find_all_correct_l_k_sequences(a_uio, l, k, all_permutations)
    Jlkcorseq = uio.get_correct_ab_sequences(a_uio, l, k)
    samelk = sorted(Jlkcorseq) == sorted([tuple(lkcorseq) for lkcorseq in Alkcorseq])
    if samelk == False:
        print("Adam's and my generated l,k correct sequences differ for the uio", a_uio)
        totalsamelk = False

    Acoef = len(Alkcorseq) - len(Acorseq)
    Jcoef = len(Jlkcorseq) - len(Jcorseq)
    J_thmcoef = uio.getThml_2_coef(a_uio, l)
    if len(set([Acoef, Jcoef, J_thmcoef])) != 1:
        print("Different coefs! for the uio", a_uio)
        print("Acoef:", Acoef)
        print("Jcoef:", Jcoef)
        print("Theorem coef:", J_thmcoef)
        break


print("Generating same correct sequences for all uio?", totalsame)
print("Generating same l,k correct sequences for all uio?", totalsamelk)
print("Getting the same l,2 coefficient?", totalsamelk)