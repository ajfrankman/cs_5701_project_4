from helpers import run_cases
from decision_tree import DecisionTree



with open('Problem4DecisionTreeData.CSV', 'r') as file:
        lines = file.readlines()
    
# Extract the header and data
headers = lines[0].strip().split(',')[1:-1]
# Data without example numbers
csv_data = [line.strip().split(',') for line in lines[1:]]
desired_vals = [True if row[-1].lower() == 'yes' else False for row in csv_data]
attributes = [[bool(int(att)) for att in row[1:-1]] for row in csv_data]  # Exclude the first column (example number) and last column (target variable)

cases = {
        'runs_per_case': 100,
        'training_ratios': [.1, .2, .3, .4, .5, .6, .7, .8, .9],
        'desired_values': desired_vals,
        'attributes': attributes,
        'headers': headers,
        'results': {}
}

run_cases(cases)
x = 2

