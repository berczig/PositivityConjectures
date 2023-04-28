import numpy as np
import time
import pickle

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam
from keras.models import load_model
from extra import PartiallyLoadable
import random



"""
class Mytest:
    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(8,  activation="relu"))
        self.model.build((None, 2))
        self.model.compile(loss="categorical_crossentropy", optimizer=SGD(learning_rate = 0.01), run_eagerly=True) #Adam optimizer also works well, with lower learning rate
    def save(self, file):
        self.model.save(file)
    def load(self, file):
        self.model = load_model(file)
        
A = Mytest()
print("weights:", A.model.layers[0].get_weights()[0])
#A.save("modelsavetest")
A.load("modelsavetest")
print("weights:", A.model.layers[0].get_weights()[0])
"""


class TestClass(PartiallyLoadable):
    def __init__(self, savevars, list):
        super().__init__(savevars)
        self.list = list
        self.x = random.random()


X = TestClass(["list"], [1,2,3])
X.lol = 33
X.save("testsave.bin")
A = TestClass(["list"], [])
A.load("testsave.bin")
print("A:", A.list)

print(vars(X))
print(vars(A))

