import numpy as np
import time
vectoraction = np.random.multinomial(1, [0.7, 0.1, 0.1, 0.1], size=1).reshape(4)
print(vectoraction)