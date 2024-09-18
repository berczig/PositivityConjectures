from functools import lru_cache
from itertools import permutations
import os
import SPC
from pathlib import Path

# COMBINATORIC FUNCTIONS
#@lru_cache(maxsize=None)
def getPermutationsOfN(n):
    return permutations(range(n))

def getKPermutationsOfN(n,k):
    return permutations(range(n), k)

def count(iterable):
    return sum((1 for _ in iterable))

def getUnusedFilepath(filepath):
    folder, filename = os.path.split(filepath)
    base_filename, extension = os.path.splitext(filename)
    newfilename = filename
    i = 1
    while newfilename in os.listdir(folder):
        newfilename = f"{base_filename}_{i:03d}{extension}"
        i += 1
    return os.path.join(folder, newfilename)

refpath = Path(SPC.__file__).parent.parent