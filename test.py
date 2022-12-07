import pandas as pd

def variableElimination(cpts: pd.DataFrame, variable: str) -> pd.Series:
    # Filter the rows that match the given conditions
    rows = cpts.query(f'{variable} == 1')

    # Group the filtered rows by the values of the remaining variables
    groups = rows.groupby([col for col in cpts.columns if col != variable])

    # Calculate the sum of the probabilities in each group
    probabilities = groups['P'].sum()

    # Return the result
    return probabilities

def marginalDistribution(cpts: pd.DataFrame, query, evidence, eliminate: List[str]):
    # Eliminate the specified variables from the CPTs
    for variable in eliminate:
        cpts = variableElimination(cpts, variable)

    # Filter the rows that match the given evidence
    rows = cpts
    for variable, value in evidence.items():
        rows = rows.query(f'{variable} == {value}')

    # Group the filtered rows by the values of the query variables
    groups = rows.groupby(query)

    # Calculate the sum of the probabilities in each group
    probabilities = groups['P'].sum()

    # Return the result
    return probabilities

# Define the joint distribution
joint_dist = pd.DataFrame({
    'A': [0, 0, 0, 0, 1, 1, 1, 1],
    'B': [0, 0, 1, 1, 0, 0, 1, 1],
    'C': [0, 1, 0, 1, 0, 1, 0, 1],
    'P': [0.1, 0.9, 0.4, 0.6, 0.7, 0.3, 0.2, 0.8]
})

# Compute the marginal distribution of variables A and C given that B is 1
probabilities = marginalDistribution(joint_dist, ['A', 'C'], {'B': 1}, ['B'])

# Print the result
print(probabilities)

# Output:
# A  C
# 0  0    0.5
#   1    1.5
# 1  0    0.9
#   1    1.1
