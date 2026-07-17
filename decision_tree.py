from math import log2
import random
import copy


class DecisionTree:
    def __init__(self, desired_values: list[bool], attributes: list[list[bool]], headers: list[str], training_size: float = .5):
        self.desired_values = desired_values
        self.attributes = attributes

        self.training_size = training_size

        self.tree = {
            'headers': headers,
            'p_node': None,
            'n_node': None
        }


    def get_entropy(self, desired_values: list[bool]) -> float:
        # Calculate the entropy of the dataset
        p_positive = sum([int(val) for val in desired_values])/len(desired_values)
        p_negative = 1 - p_positive
        if p_positive == 0 or p_negative == 0:
            return 0
        return -(p_positive * log2(p_positive) + p_negative * log2(p_negative))
   
    def get_info_gain(self, desired_values: list[bool], attribute_values: list[bool]) -> float:
        # Calculate the information gain of splitting on the given attribute
        initial_entropy = self.get_entropy(desired_values)  # Missing information
        positive_weight, negative_weight = (0, 0)
        subset_positive, subset_negative = ([], [])
        for i, att_val in enumerate(attribute_values):
            if att_val:
                subset_positive.append(desired_values[i])
                positive_weight += 1
            else:
                subset_negative.append(desired_values[i])
                negative_weight += 1
        
        positive_entropy = self.get_entropy(subset_positive) if len(subset_positive) > 0 else 0
        negative_entropy = self.get_entropy(subset_negative) if len(subset_negative) > 0 else 0

        # Calculate the remainder
        total_choices = len(desired_values)
        pos_rem = (positive_weight/total_choices) * positive_entropy
        neg_rem = (negative_weight/total_choices) * negative_entropy
        remainder = pos_rem + neg_rem #Missing information

        info_gain = initial_entropy - remainder

        return info_gain

    def _split_data(self, desired_values: list[bool], attributes: list[list[bool]], training_size: float):
        # Split data by randomly selecting training_size% of the data for training and the rest for testing
        total_size = len(self.desired_values)
        training_size = int(total_size * training_size)
        training_indices = random.sample(range(total_size), training_size)
        training_atts = [self.attributes[i] for i in training_indices]
        training_desired = [self.desired_values[i] for i in training_indices]
        testing_indices = [i for i in range(total_size) if i not in training_indices]
        testing_atts = [self.attributes[i] for i in testing_indices]
        testing_desired = [self.desired_values[i] for i in testing_indices]

        return training_atts, training_desired, testing_atts, testing_desired

    def _build_nodes(self, current_node):
        true_probability = sum([int(val) for val in current_node['desired_values']])/len(current_node['desired_values'])
        current_node['true_probability'] = true_probability
        current_node['false_probability'] = 1 - true_probability
        current_node['p_node'] = None
        current_node['n_node'] = None
        if true_probability == 1 or true_probability == 0:
            return
        
        max_info_gain_col = None  # The column with the most information gain.
        max_info_gain = -1.0  # Hold the amount of information gained used max_info_gain_col
        max_info_gain_attr = None  # The attribute with the most information gain.
        for i in range(len(current_node['headers'])): # for each attribute
            # Make a list of values for all of column i.
            attribute_values = [row[i] for row in current_node['training_atts']]
            info_gain = self.get_info_gain(desired_values=current_node['desired_values'], attribute_values=attribute_values)
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                max_info_gain_col = i
                max_info_gain_attr = current_node['headers'][i]
        
        current_node['attribute'] = max_info_gain_attr
        # Check for break conditions:
        # Pure info
        if max_info_gain <= 1e-9:
            return
        if len(current_node['headers']) == 1: # Only one attribute left
            return

        new_headers = [header for i, header in enumerate(current_node['headers']) if i != max_info_gain_col]
        p_desired = []
        p_attributes = []
        n_desired = []
        n_attributes = []

        for i, row in enumerate(current_node['training_atts']):
            att_val = row[max_info_gain_col]
            des_val = current_node['desired_values'][i]

            clean_row = row[:max_info_gain_col] + row[max_info_gain_col+1:]  # Remove the attribute that was just split on
            if att_val:
                p_desired.append(des_val)
                p_attributes.append(clean_row)
            else:
                n_desired.append(des_val)
                n_attributes.append(clean_row)

        p_node = {
            'desired_values': p_desired, 'training_atts': p_attributes,
            'headers': new_headers, 'p_node': None, 'n_node': None
        }
        if len(p_attributes) > 0:
            self._build_nodes(p_node)
        current_node['p_node'] = p_node

        n_node = {
            'desired_values': n_desired, 'training_atts': n_attributes,
            'headers': new_headers, 'p_node': None, 'n_node': None
        }
        if len(n_attributes) > 0:
            self._build_nodes(n_node)
        current_node['n_node'] = n_node


    def build_tree(self, training_size: float = None):
        training_size = training_size if training_size else self.training_size
        train_attr, train_desired, test_attr, test_desired = self._split_data(self.desired_values,self.attributes, training_size)

        self.tree['desired_values'] = train_desired
        self.tree['training_atts'] = train_attr

        self._build_nodes(self.tree)

        return test_attr, test_desired
    
    def make_decisions(self, test_attr: list[list[bool]], test_desired: list[bool]):
        # Make decisions based on the built tree and return the accuracy
        correct = 0
        for r_inx,row in enumerate(test_attr):
            # Get the decision based on current node
            current_node = self.tree
            # map row to the headers
            attr_dict = {header: value for header, value in zip(self.tree['headers'], row)}
            while current_node['p_node'] or current_node['n_node']:
                if current_node['attribute'] is None:
                    break
                attr_value = attr_dict[current_node['attribute']]
                if attr_value:
                    current_node = current_node['p_node']
                else:
                    current_node = current_node['n_node']
            # Make a decision based on the probabilities at the leaf node
            decision = current_node['true_probability'] >= 0.5
            if decision == test_desired[r_inx]:
                correct += 1
        accuracy = correct / len(test_desired)
        return accuracy