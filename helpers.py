from decision_tree import DecisionTree

def parse_file(file_path: str):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Extract the header and data
    headers = lines[0].strip().split(',')[1:-1]
    # Data without example numbers
    csv_data = [line.strip().split(',') for line in lines[1:]]
    desired_vals = [True if row[-1].lower() == 'yes' else False for row in csv_data]
    attributes = [[bool(int(att)) for att in row[1:-1]] for row in csv_data]  # Exclude the first column (example number) and last column (target variable)

    return headers, desired_vals, attributes

def run_cases(cases: dict):
    file_paths = cases.get('file_paths', None)
    file_names = cases.get('file_names', None)
    for file_name, file_path in zip(file_names, file_paths):
        headers, desired_vals, attributes = parse_file(file_path)
        cases['results'][file_name] = {}
        for ratio in cases['training_ratios']:
            train_accuracy_list = []
            test_accuracy_list = []
            for _ in range(cases['runs_per_case']):
                decision_tree = DecisionTree(desired_values=desired_vals, attributes=attributes, headers=headers, training_size=ratio)
                train_attr, train_desired, test_attr, test_desired = decision_tree.build_tree()
                train_accuracy = decision_tree.make_decisions(train_attr, train_desired)
                test_accuracy = decision_tree.make_decisions(test_attr, test_desired)
                train_accuracy_list.append(train_accuracy)
                test_accuracy_list.append(test_accuracy)
            average_train_accuracy = sum(train_accuracy_list) / len(train_accuracy_list)
            average_test_accuracy = sum(test_accuracy_list) / len(test_accuracy_list)

            cases['results'][file_name][ratio] = {
                'average_train_accuracy': average_train_accuracy,
                'average_test_accuracy': average_test_accuracy
            }
    