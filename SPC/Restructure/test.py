from keras.models import Sequential, Model

class A(Sequential, Model):
    def __init__(self):
        self = Sequential()

a = A()