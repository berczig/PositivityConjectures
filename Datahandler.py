import pandas as pd

def loadxlsx(path, trainpercentage=0.8):
    """
    Loads graph and correct data from xlsx and splits intro training|testing
    """
    df = pd.read_excel(path)
    values = df.values
    print("labes from xlsl:", values[0])
    Graphdata = values[1:, :8].astype(float) # data describing the graph
    Correctnessdata = values[1:, 8:10].astype(float) # "only l3 correct" and "only l+3 correct"
    n = len(Graphdata)
    K = int(trainpercentage*n)
    return Graphdata[:K], Graphdata[K:], Correctnessdata[:K], Correctnessdata[K:]

if __name__ == "__main__":
    X, X2,Y, Y2 = loadxlsx('chrom53.xlsx', 0.1)
    print(X)
    print(Y)