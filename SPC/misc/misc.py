from functools import lru_cache
from itertools import permutations
import time

# COMBINATORIC FUNCTIONS
#@lru_cache(maxsize=None)
def getPermutationsOfN(n):
    return permutations(range(n))

def count(iterable):
    return sum((1 for _ in iterable))