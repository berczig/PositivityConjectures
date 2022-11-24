import os
import pandas as pd
from sklearn.utils import shuffle
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np


def loadxlsx(path, trainpercentage=0.9):
    """
    Loads graph and correct data from xlsx and splits intro training|testing
    """
    df = pd.read_excel(path)
    values = df.values
    print("labels from xlsl:", values[0])
    UIOdata = values[1:, :9].astype(float)  # data describing the graph
    Correctnessdata = values[1:, 9:11].astype(float)  # "only l3 correct" and "only l+3 correct"
    n = len(UIOdata)
    K = int(trainpercentage * n)
    return UIOdata[:K], UIOdata[K:], Correctnessdata[:K], Correctnessdata[K:]

def SplitCSV(path,percentage):
    dforiginal = pd.read_csv(path)
    df = shuffle(dforiginal)
    low = 1  # Initial Lower Limit
    high = int(percentage*len(df))  # Initial Higher Limit
    df_train = df[low:high]  # subsetting DataFrame based on index
    df_test = df[high:]
    df_train.to_csv(path[0:-4]+"train.csv", index=False)  # output file
    df_test.to_csv(path[0:-4]+"test.csv",index=False)  # output file

class CustomDataset(Dataset):
    def __init__(self, csv_path, difference):
        fields = ["i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8", "i9", "only l3 correct", "only l+3 correct",
                  "both correct", "only l+3 correct (1)", "only l+3 correct (2)", "only l+3 correct (3)",
                  "only l+3 correct (4)"];
        df = pd.read_csv(csv_path, sep=";", header=None, names=fields)
        values = df.values
        if difference == False:
            self.UIO_labels = torch.from_numpy(values[1: , 9:11].astype(np.float32))/3000
        else:
            self.UIO_labels = torch.from_numpy(values[1:, 9:10].astype(np.float32)-values[1:, 10:11].astype(np.float32)) / 3000
        self.UIO_data = torch.from_numpy(values[1: , :9].astype(np.float32))

    def __len__(self):
        return len(self.UIO_labels)

    def __getitem__(self, idx):
        return self.UIO_data[idx], self.UIO_labels[idx]


    def loadcsv(path, trainpercentage=0.9):
        """
        Loads graph and correct data from xlsx and splits intro training|testing
        """
        fields = ["i1","i2","i3","i4","i5","i6","i7","i8","i9","only l3 correct","only l+3 correct","both correct","only l+3 correct (1)","only l+3 correct (2)","only l+3 correct (3)","only l+3 correct (4)"];
        df = pd.read_csv(path, sep=";", header = None, names=fields)
        values = df.values
        print("labels from xlsl:", values[0])
        UIOdata = values[1: , :9].astype(float) # data describing the graph
        Correctnessdata = values[1: , 9:11].astype(float) # "only l3 correct" and "only l+3 correct"
        n = len(UIOdata)
        K = int(trainpercentage*n)
        return UIOdata[:K], UIOdata[K:], Correctnessdata[:K], Correctnessdata[K:]


class CustomDatasetTest(Dataset):
    def __init__(self):
        self.n = 1000
        self.W = torch.tensor([
            [2.0,5.0,7.0,2.0,0.0,0.0,8.0,5.0,-1.0],
            [1.0,2.0,3.0,7.0,6.0,-4.0,-3.0,0.0,9.0]
        ]).T
        print("Weights shape:", self.W.shape)
        self.X = torch.rand(self.n, 9)
        self.Y = torch.matmul(self.X, self.W)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


    def loadcsv(path, trainpercentage=0.9):
        """
        Loads graph and correct data from xlsx and splits intro training|testing
        """
        fields = ["i1","i2","i3","i4","i5","i6","i7","i8","i9","only l3 correct","only l+3 correct","both correct","only l+3 correct (1)","only l+3 correct (2)","only l+3 correct (3)","only l+3 correct (4)"];
        df = pd.read_csv(path, sep=";", header = None, names=fields)
        values = df.values
        print("labels from xlsl:", values[0])
        UIOdata = values[1: , :9].astype(float) # data describing the graph
        Correctnessdata = values[1: , 9:11].astype(float) # "only l3 correct" and "only l+3 correct"
        n = len(UIOdata)
        K = int(trainpercentage*n)
        return UIOdata[:K], UIOdata[K:], Correctnessdata[:K], Correctnessdata[K:]





if __name__ == "__main__":
    X = CustomDatasetTest()
    #SplitCSV("Chrom63.csv",0.8)
