from decision_tree import DecisionTree


def run_cases(cases: dict):
    for ratio in cases['training_ratios']:
        accuracy_list = []
        for _ in range(cases['runs_per_case']):
            decision_tree = DecisionTree(desired_values=cases['desired_values'], attributes=cases['attributes'], headers=cases['headers'], training_size=ratio)
            test_attr, test_desired = decision_tree.build_tree()
            accuracy = decision_tree.make_decisions(test_attr, test_desired)
            accuracy_list.append(accuracy)
        average_accuracy = sum(accuracy_list) / len(accuracy_list)
        cases['results'][ratio] = average_accuracy
    