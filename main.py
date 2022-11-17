from Datahandler import *
from LinearTransformer import *
import matplotlib.pyplot as plt

trainX, testX, trainY, testY = loadxlsx("Chrom63.csv", 0.8)

plt.plot(trainY[:, 0], "g", trainY[:, 1], "r")
plt.xlabel("Graph ID")
plt.legend(["only l3 correct", "only l+3 correct"])
plt.show()

LT = LinearTransformer(trainX, trainY, normed=True)
print("Insample error:", LT.getScore(trainX, trainY))
print("Outsample error:", LT.getScore(testX, testY))

Yhat = LT.map(trainX)
coeffs = trainY[:, 0]-trainY[:, 1]
coeffs_pred = Yhat[:, 0] - Yhat[:, 1]
plt.plot(coeffs, "g", coeffs_pred, "r")
plt.xlabel("Graph ID")
plt.legend(["real coeffs", "predicted coeffs(insample)"])
plt.show()


Yhat = LT.map(testX)
coeffs = testY[:, 0]-testY[:, 1]
coeffs_pred = Yhat[:, 0] - Yhat[:, 1]
plt.plot(coeffs, "g", coeffs_pred, "r")
plt.xlabel("Graph ID")
plt.legend(["real coeffs", "predicted coeffs(outsample)"])
plt.show()