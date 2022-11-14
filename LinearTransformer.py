import numpy as np


class Normalizer():
    """
    Helper class for normalizing data 
    """

    def __init__(self, X, verbose=False):
        """
        calculates the mean and standard deviation of X, data as rows in X
        """

        self.mean = np.mean(X, axis=0) # take mean accros the i'th coordinate
        self.std = np.std(X)

        if verbose:
            print("Data:", X)
            print("Mean:", self.mean)
            print("standard deviation:", self.std)

    def normalize(self, X):
        """
        normalizes X, data as rows in X
        """
        return (X-self.mean)/self.std
    
    def denormalize(self, X):
        """
        denormalizes X, data as rows in X
        """
        return self.std*X + self.mean
        

class LinearTransformer():

    def __init__(self, X, Y, normed=False):
        """
        Data in X,Y as rows
        """
        self.normed = normed
        if normed:
            self.inputNormalized = Normalizer(X)
            self.outputNormalized = Normalizer(Y)
            
            self.matrix = self.solveMatrixLSEXWY(
                    self.inputNormalized.normalize(X), 
                    self.outputNormalized.normalize(Y))
        else:
            self.matrix = self.solveMatrixLSEXWY(X, Y)


    def solveMatrixLSEXWY(self, X,Y): 
        """
        Finds W s.t. XW =~ Y
        returns argmin_W sum_i |W^T*X[i,:]^T - Y[i,:]^T|^2
        X is n x m
        W is m x k
        Y is n x k
        """
        return np.dot(np.linalg.pinv(X), Y)

    def map(self, X):
        """
        multiplies X by a matrix
        """
        if self.normed:
            return self.outputNormalized.denormalize(self.inputNormalized.normalize(X)@self.matrix)
        else:
            return X@self.matrix

    def getScore(self, X, Y):
        """
        maps X and compares the result to Y (using Frobenius norm)
        """
        return np.linalg.norm(self.map(X)-Y)

if __name__ == "__main__":
    # let's see if it can determine the correct linear transformation
    W = np.array([
        [1,2,6],
        [-2,6,7],
        [1,7,5]
    ])

    X = np.array([
        [1,1,3],
        [2,2,0],
        [-1,-3,4],
        [5,7,3]
    ])
    Y = X@W
    print("Without normalization:")
    LT = LinearTransformer(X, Y)
    print("Score:", LT.getScore(X,Y))
    print(LT.matrix)

    print("With normalization:")
    LT = LinearTransformer(X, Y, True)
    print("Score:", LT.getScore(X,Y))
    print(LT.matrix)