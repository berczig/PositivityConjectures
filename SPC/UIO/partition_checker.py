def getPartitions():
    return [(4,1,1), (3,3,2,1), (3,2,1), (4,3,2), (4,2,1,1)]

def checkPartition(partition):
    # not non-increasing
    if tuple(sorted(partition, key = lambda x: -x)) != partition:
        print(partition,"IS NOT A PARTITION")

def permutePartition(partition):
    print("check for partition:", partition)
    if len(partition) == 3:
        n,k,l = partition
        checkPartition((n+k+l,))
        checkPartition(partition) 
        checkPartition((n+l,k))
        checkPartition((n+k,l))
        checkPartition((l+k,n))

    elif len(partition) == 4:
            a,b,c,d = partition
            checkPartition((a,b,c,d))
            checkPartition((a+b,c,d))
            checkPartition((a+c,b,d)) 
            checkPartition((a+d,b,c)) 
            checkPartition((b+c,a,d)) 
            checkPartition((b+d,a,c)) 
            checkPartition((c+d,a,b)) 
            checkPartition((a+b,c+d)) 
            checkPartition((a+c, b+d)) 
            checkPartition((a+d, b+c)) 
            checkPartition((a+b+c, d)) 
            checkPartition((a+b+d, c)) 
            checkPartition((a+c+d,b)) 
            checkPartition((b+c+d,a)) 
            checkPartition((a+b+c+d,))
    print()

for part in getPartitions():
    permutePartition(part)