#!/usr/bin/python
import numpy
import csv
from csv import reader
from matplotlib import pyplot
import sys
import math

def load_csv(filename):
    csv_reader = reader(open(filename, 'rt'))
    x = list(csv_reader)
    result = numpy.array(x)
    result = numpy.delete(result, 0, 0)
    col = result[:,6]
    result = numpy.delete(result, 6, 1)
    result = numpy.hstack((result,col))
    #print result
    return result

def load_csv1(filename):
    csv_reader = reader(open(filename, 'rt'))
    x = list(csv_reader)
    result = numpy.array(x)
    result = numpy.delete(result, 0, 0)
    #print result
    return result

def unique(result):
    answer = []
    num_rows,num_col = result.shape
    for i in range(num_cols-1):
        for k in range(num_row):
            if result[k][i] not in answer[i]:
                answer[i].append(result[k][i])
    return answer

def wholeH(result):
	count_leave = 0
	count_stay = 0
	for i in range(num_rows):
		if result[9][i] == '0':
			count_stay += 1;
		else:
			count_leave +=1
	pos = (count_leave*1.0)/(count_stay+count_leave)
	neg = (count_stay*1.0)/(count_stay+count_leave)
	p = -1*(pos*math.log(pos,2) + neg*math.log(neg,2))
	return p

def to_terminal(node):
    num_rows, num_cols = node.shape
    count = 0
    count1 = 0
    for i in range(num_rows):
        if int(node[i][9]) == 1:
            count +=1
        if int(node[i][9]) == 0:
            count1 +=1
    if count >= count1:
        return 1
    else:
        return 0

def get_tree(node):
    left, right = node['groups']
    del(node['groups'])
    if (not left):
        node['left'] = to_terminal(right)
    if (not right):
        node['right'] = to_terminal(left)
    if len(left) > 0 and len(left) <= 100:
        node['left'] = to_terminal(left)
    elif len(left) > 100:
        node['left'] = gainfunction(left)
        get_tree(node['left'])
    if len(right) > 0 and len(right) <= 100:
        node['right'] = to_terminal(right)
    elif len(right) > 100:
        node['right'] = gainfunction(right)
        get_tree(node['right'])

def test(node, row):
    num_rows,num_col = result.shape
    if node['index'] < 7:
        if row[node['index']] <= node['value']:
            if isinstance(node['left'], dict):
                return test(node['left'], row)
            else:
                print (node['left'])
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return test(node['right'], row)
            else:
                print (node['right'])
                return node['right']
   else:
       if row[node['index']] != node['value']:
           if isinstance(node['left'], dict):
               return test(node['left'], row)
           else:
               return node['left']
       else:
           if isinstance(node['right'], dict):
               return test(node['right'], row)
           else:
               return node['right']

def split (result, index, val):
    left = []
    right = []
    num_rows,num_col = result.shape
    if index < 7:
        for j in range(num_rows):
            if result[j][index] <= val:
                left.append(result[j])
            else:
                rigth.append(result[j])
    else:
        for j in range(num_rows):
            if result[j][index] != val:
                left.append(result[j])
            else:
                rigth.append(result[j])
    left = numpy(left)
    right = numpy(right)
    groups = left, right
    b_index, b_value, b_groups = index, val, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}

def gainfunction(result):
    H = wholeH(result)
    answer = unique(result)
    num_rows,num_col = result.shape
    energy_gain = -1
    index = 0
    val = 0
    for k in range(num_col-2):
    	count1 = 0
    	count2 = 0
    	count3 = 0
    	count4 = 0
    	pa = 0
    	pb = 0
        for j in range(len(answer[k])):
            for i in range(num_rows):
                if float(result[i][k]) <= float(answer[k][j]):
                    count1 += 1
                if float(result[i][k]) <= float(answer[k][j]) and result[i][9] == '1':
                    count2 += 1
                if float(result[i][k]) > float(answer[k][j]):
                    count3 += 1
                if float(result[i][k]) > float(answer[k][j]) and result[i][9] == '1':
                    count4 += 1
            p3 = (count2*1.0)/count1
            p4 = (count1-count2)/(count1*1.0)
            p5 = (count4*1.0)/count3
            p6 = (count3-count4)/(count3*1.0)
            p7 = count1 + count3
            if (p3 == 0 or p3 ==1):
            	pa = 0
            else:
            	pa = -1*(p3*math.log(p3,2) + p4*math.log(p4,2))
            if (p5 == 0 or p5 ==1):
            	pb = 0
            else:
            	pb = -1*(p5*math.log(p5,2) + p6*math.log(p6,2))
            p1 = ((count1*1.0)/p7)*pa + ((count3*1.0)/p7)*pb
            inf_gain= H - p1
            if inf_gain > energy_gain:
                energy_gain = inf_gain
                index = i
                val = float(answer[i][j])
    i = 7
    while  i < num_col:
        count1 = 0
    	count2 = 0
    	count3 = 0
    	count4 = 0
    	pa = 0
    	pb = 0
        for j in range(len(answer[i])):
            for k in range(num_rows):
                if result[k][i] != answer[i][j]:
                    count1 += 1
                if result[k][i] != answer[i][j] and result[i][9] == '1':
                    count2 += 1
                if result[k][i] == answer[i][j]:
            		count3 += 1
                if result[k][i] == answer[i][j] and result[i][9] == '1':
            		count4 += 1
            if (count1 == 0 and count3 != 0):
                pa = 0
                p5 = (count4*1.0)/count3
                p6 = (count3-count4)/(count3*1.0)
                p7 = count1 + count3
                if (p5 == 0 or p5 ==1):
                	pb = 0
                else:
                	pb = -1*(p5*math.log(p5,2) + p6*math.log(p6,2))

            elif (count1 == 0 and count3 == 0):
                pa = 0
                pb = 0
            elif (count1 != 0 and count3 == 0):
                pb = 0
                p3 = (count2*1.0)/count1
                p4 = (count1-count2)/(count1*1.0)
                p7 = count1 + count3
                if (p3 == 0 or p3 ==1):
                	pa = 0
                else:
                	pa = -1*(p3*math.log(p3,2) + p4*math.log(p4,2))
            else:
                p3 = (count2*1.0)/count1
                p4 = (count1-count2)/(count1*1.0)
                p5 = (count4*1.0)/count3
                p6 = (count3-count4)/(count3*1.0)
                p7 = count1 + count3
                if (p3 == 0 or p3 ==1):
            	       pa = 0
                else:
            	       pa = -1*(p3*math.log(p3,2) + p4*math.log(p4,2))
                if (p5 == 0 or p5 ==1):
            	       pb = 0
                else:
            	       pb = -1*(p5*math.log(p5,2) + p6*math.log(p6,2))
            p1 = ((count1*1.0)/p7)*pa + ((count3*1.0)/p7)*pb
            inf_gain= H - p1
            if inf_gain > energy_gain:
                energy_gain = inf_gain
                index = i
                val = answer[i][j]
        i += 1
    return split(result,index,val)

dataset_train = sys.argv[1]
dataset_test = sys.argv[2]
result = load_csv(dataset_train)
result1 = load_csv1(dataset_test)
root = gainfunction(result)
tree = get_tree(root)
num_row, num_col = result1.shape

test(root)
