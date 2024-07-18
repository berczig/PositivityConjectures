from gurobipy import Model, GRB

def solve_subset_sum(matrix, targets):
    n_rows, n_cols = len(matrix), len(matrix[0])
    
    # Create a new model
    m = Model("subset_sum")
    
    # Add decision variables
    x = m.addVars(n_cols, vtype=GRB.BINARY, name="x")
    
    # Add constraints for each row to match the target sum
    for i in range(n_rows):
        m.addConstr(sum(matrix[i][j] * x[j] for j in range(n_cols)) == targets[i], f"target_{i}")
    
    # Since it's a feasibility problem, we don't have an explicit objective to optimize
    m.setObjective(0, GRB.MAXIMIZE)
    
    # Optimize model
    m.optimize()
    
    # Check if a solution exists
    if m.status == GRB.OPTIMAL or m.status == GRB.FEASIBLE:
        solution = [j for j in range(n_cols) if x[j].X > 0.5]  # Extracting the solution
        print("Solution found: Columns included in the subset are:", solution)
    else:
        print("No solution exists.")

# Example usage
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # nxk table of integers
targets = [6, 15, 24]  # Set of target sums T={t_1,...,t_n}
solve_subset_sum(matrix, targets)


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
    
    solve_subset_sum(matrix, targets)

# Adjust the path to the file as necessary
filename = "SPC/Saves,Tests/subsetsum/subsetsum3_2_1_symmetric.txt"
read_and_solve_with_first_column_as_target(filename)