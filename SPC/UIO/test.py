import numpy as np
import time

a = np.ones(10)
a[0] = -2
print((a>0).any())
print((a>0).all())
#print(a)
N = 10**4
t = time.time()
for i in range(N):
    #for x in a:
    #    if x < 0:
    #        print("y9")
    #if not (a>0).all():
        #pass
    if  (a<0).any():
        pass
print(time.time()-t)