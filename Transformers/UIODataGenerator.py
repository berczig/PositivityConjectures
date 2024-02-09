# Handling various forms of GPT2 training data created from the Stanley coefficients
# n = length of UIO

#Default version: GPT learning the coefficients (l,3)
# GPT training on pairs (X, Y):
#     input: a UIO X as a length n sorted integer vector 
#     output: a length n vector Y whose ith entry is 
#                 Y[i]=coeff(3,i-3) for i>2 and Y[i]=0 for i=0,1,2

from eshers import generate_all_uios
import torch

def getUIOVector(n):
    A = generate_all_uios(n)
    return torch.stack(A)

def getCoeffVector(n)


    #A = [[0, 0, 1, 1, 2, 3]]
    for encod in A:
        uio  = UIO(encod)  