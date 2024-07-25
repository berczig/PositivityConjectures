
from gurobipy import Model, GRB
import numpy as np

from SPC.Restructure.GlobalUIODataPreparer import GlobalUIODataPreparer
from SPC.Restructure.cores.EscherCoreGeneratorBasic import EscherCoreGeneratorBasic
from SPC.Restructure.cores.EscherCoreGeneratorTrippleSymmetric import EscherCoreGeneratorTrippleSymmetric

def solve_subset_sum(matrix, targets, max_solutions=1):
    n_rows, n_cols = len(matrix), len(matrix[0])
    solutions = []

    # Create a new model
    m = Model("subset_sum")

    # Add decision variables
    x = m.addVars(n_cols, vtype=GRB.BINARY, name="x")

    # Add constraints for each row to match the target sum
    for i in range(n_rows):
        m.addConstr(sum(matrix[i][j] * x[j] for j in range(n_cols)) == targets[i], f"target_{i}")

    # Since it's a feasibility problem, we don't have an explicit objective to optimize
    m.setObjective(0, GRB.MAXIMIZE)

    # Configure the model to find multiple solutions
    m.setParam(GRB.Param.PoolSearchMode, 2)
    m.setParam(GRB.Param.PoolSolutions, max_solutions)

    # Optimize model
    m.optimize()

    # Check if solutions were found
    if m.status == GRB.OPTIMAL or m.status == GRB.FEASIBLE:
        solution_count = m.SolCount
        for s in range(min(solution_count, max_solutions)):
            m.setParam(GRB.Param.SolutionNumber, s)
            solution = [j for j in range(n_cols) if x[j].Xn > 0.5]
            solutions.append(solution)
            #print(f"Solution {s}: {solution}")
            #print('Length:', len(solution))

    return solutions

def read_and_solve_with_first_column_as_target(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        # Initialize an empty matrix and an empty list for targets
        matrix = []
        targets = []
        for line in lines:
            numbers = list(map(int, line.split(',')))
            # The first number in each line is the target sum for that row
            targets.append(numbers[0])
            # The rest of the numbers form part of the matrix
            matrix.append(numbers[1:])
        solutions = solve_subset_sum(matrix, targets)
    
    return targets, matrix, solutions


# Adjust the path to the file as necessary
filename = "SPC/Saves,Tests/subsetsum/subsetsum3_2_1_symmetric.txt"
targets, matrix, solutions = read_and_solve_with_first_column_as_target(filename)

print("Solutions:", solutions)

# Print the intersection of the solutions

intersection = set(solutions[0])
for solution in solutions[1:]:
    intersection = intersection.intersection(set(solution))

print("Intersection:", intersection)

#solutions.append(list(intersection))

partition = (4,2)
#core_generator_type = "EscherCoreGeneratorTripple" # EscherCoreGeneratorBasic  EscherCoreGeneratorTripple

uio_length = sum(partition)
# core_data_type

# 1) get Training Data
Preparer = GlobalUIODataPreparer(uio_length)
#Preparer.computeTrainingData(partition, core_generator_type)
Preparer.loadTrainingData("SPC/Saves,Tests/Trainingdata/partition_3_2_1_symmectric_core_9_7_2024.bin")

from sklearn.tree import DecisionTreeClassifier, export_text


for solution in solutions:
    print("SOLUTION")
    
    solutioncorereps = []
    notsolutioncorereps = []
    X = []
    y = []
    for corerepID, corerep in enumerate(Preparer.coreRepresentationsCategorizer):
        if corerep != 'GOOD':
            X.append(corerep)
            print("corerep meaning:", EscherCoreGeneratorTrippleSymmetric.convertCorerepToText(corerep, partition), "\n")
            if corerepID in solution:
                solutioncorereps.append(corerep)
                y.append(1)
            else:
                notsolutioncorereps.append(corerep)
                y.append(0)

    print("Length of solution core representations:", len(solutioncorereps))
    print("Length of not solution core representations:", len(notsolutioncorereps))
    print('y', y)

    # Decision tree classifier


    # Convert the data to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Train the decision tree
    clf = DecisionTreeClassifier(splitter="random", max_depth=3)
    clf.fit(X, y)

    # Extract the logical expression
    tree_rules = export_text(clf, feature_names=[f"v{i}" for i in range(len(X[0]))])
    print(tree_rules)

    # Calculate and print the efficiency of the tree on the training data
    accuracy = clf.score(X, y)
    print(f"Efficiency (Accuracy) on Training Data: {accuracy*100:.2f}%")

    # Calculate the 

    # Print how many points are  misclassified

    predictions = clf.predict(X)
    misclassified = np.where(predictions != y)[0]
    print(f"Number of misclassified points: {len(misclassified)}")

    # Print the misclassified points
    for i in misclassified:
        print(f"Point {i} is misclassified as {predictions[i]}")

    
    # Make a 2d numpy array from matrix

    matrix = np.array(matrix)

    print(matrix.shape)

    #Add the columns of matrix indexed by entries of 'intersection' 

    sum_of_misclassified = np.sum(matrix[:, list(misclassified)], axis=1)

    print("Sum of misclassified:", sum_of_misclassified)


    # Subtract the sum of the intersection from the targets

    #targets = np.array(targets)

    #remainder = targets - sum_of_misclassified

    #print("Remainder:", remainder)
    

    

    

    




