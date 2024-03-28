from functools import lru_cache
from itertools import permutations

# COMBINATORIC FUNCTIONS
@lru_cache(maxsize=None)
def getPermutationsOfN(n):
    return list(permutations(range(n)))