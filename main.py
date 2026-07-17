from helpers import run_cases, parse_file
from decision_tree import DecisionTree




# headers, desired_vals, attributes = parse_file('no_safe_situation.CSV')
data_file_names = [
    'Problem4DecisionTreeData.csv',
    'minus_1_col.csv',
    'minus_2_col.csv',
]
data_file_folder = 'data/'
file_paths = [data_file_folder + file_name for file_name in data_file_names]
cases = {
        'file_paths': file_paths,
        'file_names': data_file_names,
        'runs_per_case': 100,
        'training_ratios': [.1, .2, .3, .4, .5, .6, .7, .8, .9],
        'results': {}
}

run_cases(cases)
# Create a table. For each file, print the average training and testing accuracy for each training ratio.
# CSV with two columns for each file, one for training accuracy and one for testing accuracy. Each row is a training ratio.
csv_str = ''
# Add headers:
header_str = ','
for file_name in cases['file_names']:
    header_str += f'{file_name},,'
''
csv_str = header_str + '\n' + 'Training Ratio,'
for file_name in cases['file_names']:
    csv_str += 'Training Accuracy, Testing Accuracy, '
csv_str += '\n'
for ratio in cases['training_ratios']:
    row = [f'{ratio}']
    for file_name in cases['file_names']:
        result = cases['results'][file_name][ratio]
        row.append(f"{result['average_train_accuracy']:.4f}")
        row.append(f"{result['average_test_accuracy']:.4f}")
    csv_str += ', '.join(row) + '\n'

with open('results.csv', 'w') as f:
    f.write(csv_str)
x = 2

