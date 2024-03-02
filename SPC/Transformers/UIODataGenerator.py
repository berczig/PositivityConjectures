# Handling various forms of GPT2 training data created from the Stanley coefficients
# n = length of UIO

#Default version: GPT learning the Stanley coefficient corresponding to partiton lambda of n
#GPT training on pairs (X, Y):
#     input: a UIO X as a length n sorted integer vector 
#     output: an integer Y, the lambda-coefficient. 

from eschers import UIO
from eschers import generate_all_uios
import torch

def getUIOTrainingVectors(n):
    coeffs = []
    Xuio = torch.tensor(generate_all_uios(n))
    for encod in Xuio:
        uio = UIO(encod) 
        coeffs.append(uio.getCoeffientByEscher(n-2,2,0)) 
    Yuio = torch.tensor(coeffs).unsqueeze(1)
    return Xuio, Yuio


    #Second version: GPT learning the coefficients (l,3)
    # GPT training on pairs (X, Y):
    #     input: a UIO X as a length n=l+3 sorted integer vector 
    #     output: a length n vector Y whose ith entry is 
    #                 Y[i]=coeff(3,i-3) for i>2 and Y[i]=0 for i=0,1,2

    def getUIOTrainingVectors(n):
        coeffs = []
        A = generate_all_uios(n)
        Xuio = torch.stack(A)
        for encod in A:
            uio = UIO(encod) 
            coeffs.append(uio.getCoeffientByEscher(n-3,2,1)) 
        Yuio = torch.stack(coeffs)
        return Xuio, Yuio

#getUIOTrainingVectors(6)