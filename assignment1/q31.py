import sys
from csv import reader
import numpy as np
from math import log
#import pandas as pd

attribute = { 0: 'satisfaction_level', 1: 'last_evaluation', 2: 'number_project', 3: 'average_montly_hours', 4: 'time_spend_company',
5: 'Work_accident', 6: 'left', 7 : 'promotion_last_5years', 8: 'sales', 9: 'salary'}

def load_csv(filename):
    dataset = np.genfromtxt(filename, delimiter = ',', names=True, dtype=('<f8', '<f8', '<i8','<i8','<i8','<i8','<i8','<i8','S11','S6'))
    label = dataset['left']
    '''
    #Check if any value is missing
    s = pd.Series(column)
    print pd.isnull(s).sum()
    '''
    #print label
    #print dataset.dtype
    return dataset, label

def test_load_csv(filename):
    file = open(filename, "rt")
    lines = reader(file)
    testdata = list(lines)
    testdata.pop(0)
    return testdata

def entropy(ls):
    pos = 0; neg = 0
    for row in ls:
        if row[6] == 1:
            pos = pos + 1
        else:
            neg = neg + 1
    total = len(ls)
    if total != 0:
        prob_pos = pos/(total * 1.0)
        prob_neg = neg/(total * 1.0)
    else:
        prob_pos = 0
    if prob_pos == 0 or prob_neg == 0:
        return 0
    else:
        return (-prob_pos * log(prob_pos, 2) -prob_neg * log(prob_neg, 2))

def information_gain(groups, label):
    n_instances = float(sum([len(group) for group in groups]))
    sum_lists = []
    for group in groups:
        sum_lists = group + sum_lists
    entropy_sample = entropy(sum_lists)
    gain = entropy_sample
    for group in groups:
        entropy_part = entropy(group)
        gain = gain - ((len(group))/(len(sum_lists) * 1.0)) * entropy_part
    return gain

def test_split(index, value, dataset):
    if index <= 7:
        left, right = list(), list()
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
    else:
        left, right = list(), list()
        for row in dataset:
            if row[index] == value:
                left.append(row)
            else:
                right.append(row)
    return left, right


def get_split(dataset):
    #label = dataset[attribute[6]]
    #print len(label)
    #print label
    itr = (index for index in range(len(dataset[0])) if index != 6)
    b_index, b_value, b_gain, b_groups = -1, 0, -1, None
    for index in itr:
        temp = list()
        for row in dataset:
            if row[index] in temp:
                continue
            temp.append(row[index])
            groups = test_split(index, row[index], dataset)
            gain = information_gain(groups, label)
            if gain > b_gain:
                b_index, b_value, b_gain, b_groups = index, row[index], gain, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}

def leave_node(group):
    outcomes = [row[6] for row in group]
    return max(set(outcomes), key=outcomes.count)

def split(node, min_size):

    left, right = node['groups']
    del(node['groups'])
    #process left child

    if not left or not right:
		node['left'] = node['right'] = leave_node(left + right)
		return

    if len(left) <= min_size:
        node['left'] = leave_node(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], min_size)

    #process right chiid
    if len(right) <= min_size:
        node['right'] = leave_node(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], min_size)

def build_tree(dataset, min_size):
    root = get_split(dataset)
    split(root, min_size)
    return root

def predict(node, row):
    if node['index'] < 7:
        if float(row[node['index']]) < node['value']:
            if isinstance(node['left'], dict):
                return predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return predict(node['right'], row)
            else:
                return node['right']

    elif node['index'] == 7:
        if int(row[node['index'] - 1]) < node['value']:
            if isinstance(node['left'], dict):
                return predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return predict(node['right'], row)
            else:
                return node['right']

    else:
        if str(row[node['index'] - 1]) == node['value']:
            if isinstance(node['left'], dict):
                return predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return predict(node['right'], row)
            else:
                return node['right']

def decision_tree(dataset, test, min_size):
    tree = build_tree(dataset, min_size)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return (predictions)

def print_predictions(predictions):
    for p in predictions:
        print p

filename = sys.argv[1]
testfile = sys.argv[2]
dataset,label = load_csv(filename)
testdata = test_load_csv(testfile)

min_size = 0.01 * len(dataset)

predictions = decision_tree(dataset, testdata, min_size)
print_predictions(predictions)
