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
    return result, col

answer = []
answer1 = []
answer2 = []
answer3 = []
answer4 = []
answer5 = []
answer6 = []
answer7 = []
answer8 = []
dataset_train = 'datasets/q3/train.csv'
dataset_test = 'datasets/q3/test.csv'
result , label = load_csv(dataset_train)
result1, label1 = load_csv(dataset_test)
print (result)
num_rows, num_cols = result.shape

for i in range(num_rows):
	if result[i][0] not in answer:
		answer.append(result[i][0])
print (answer)
for i in range(num_rows):
        if result[i][1] not in answer1:
                answer1.append(result[i][1])
print (answer1)
for i in range(num_rows):
        if result[i][2] not in answer2:
                answer2.append(result[i][2])
print (answer2)
for i in range(num_rows):
        if result[i][3] not in answer3:
                answer3.append(result[i][3])
print (answer3)
for i in range(num_rows):
        if result[i][4] not in answer4:
                answer4.append(result[i][4])
print (answer4)
for i in range(num_rows):
        if result[i][5] not in answer5:
                answer5.append(result[i][5])
print (answer5)
for i in range(num_rows):
        if result[i][6] not in answer6:
                answer6.append(result[i][6])
print (answer6)

for i in range(num_rows):
    if result[i][7] not in answer7:
        answer7.append(result[i][7])
print (answer7)

for i in range(num_rows):
        if result[i][8] not in answer8:
                answer8.append(result[i][8])
print (answer8)

######################## Calculate entropy for all cases ############
def wholeH(label):
	count_leave = 0
	count_stay = 0
	for i in range(num_rows):
		if label[i] == '0':
			count_stay += 1;
		else:
			count_leave +=1
	pos = (count_leave*1.0)/(count_stay+count_leave)
	neg = (count_stay*1.0)/(count_stay+count_leave)
	p = -1*(pos*math.log(pos,2) + neg*math.log(neg,2))
	return p
################################# Calculate information gain for all satisfaction level #############
def countimpormationgain(col,H,val):
	num_rows = len(col)
	count1 = 0
	count2 = 0
	count3 = 0
	count4 = 0
	pa = 0 
	pb = 0
	for i in range(num_rows):
    		if float(col1[i]) <= val:
			count1 += 1
    		if float(col1[i]) <= val and label[i] == '1':
			count2 += 1
    		if float(col1[i]) > val:
			count3 += 1
    		if float(col1[i]) > val and label[i] == '1':
			count4 += 1
	if (count3 ==0 or count1 ==0):
		return 0
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
	return inf_gain

H = wholeH(label)
print ("############SATISFACTION LEVEL########################################")
col1 = result[:,0]
c = 0
inf_gain = 0
pa = 0
pb =0
for i in range(len(answer)):
	print len(answer)
	inf_gain = countimpormationgain(col1, H, float(answer[i]))
	if inf_gain  > c:
		c = inf_gain
		d = answer[i]
		#pa_new = pa
		#pb_new = pb
#print(pa_new)
#print(pb_new)
print(d)
print (c)
print ("################WORK ACCIDENTS####################################")
col1 = result[:,5]
inf_gain = countimpormationgain(col1, H, 0)
print(inf_gain)
print ("###########PROMOTION LAST FIVE YEAR#########################################")
col1 = result[:,6]
inf_gain = countimpormationgain(col1, H, 0)
print(inf_gain)
print ("###########NUMBER OF PROJECTS#########################################")
col1 = result[:,2]
c = 0
for i in range(len(answer2)):
	inf_gain = countimpormationgain(col1, H, float(answer2[i]))
	if inf_gain > c:
		c = inf_gain
		d = answer2[i]
print(d)
print(c)
print ("###########LAST EVALUATION#########################################")
col1 = result[:,1]
c = 0
for i in range(len(answer1)):
	inf_gain = countimpormationgain(col1, H, float(answer1[i]))
	if inf_gain > c:
		c = inf_gain
                d = answer1[i]
print(d)
print(c)

print ("###########AVERAGE MONTHLY HOURS#########################################")
col1 = result[:,3]
c = 0
for i in range(len(answer3)):
        inf_gain = countimpormationgain(col1, H, float(answer3[i]))
        if inf_gain > c:
                c = inf_gain
                d = answer3[i]
print(d)
print(c)
print("#########SALARY###########################################")
col1 = result[:,8]
count1 = 0
count2 = 0
count3 = 0
count4 = 0
count5 = 0
count6 = 0
for i in range(num_rows):
    if col1[i] == "low":
        count1 += 1
    if col1[i] == "low" and label[i] == '0':
        count2 +=1
    if col1[i] == "medium":
        count3 +=1
    if col1[i] == "medium" and label[i] == '0':
        count4 +=1
    if col1[i] == "high":
        count5 +=1
    if col1[i] == "high" and label[i] == '0':
        count6 +=1
p3 = (count2*1.0)/count1
p4 = (count1-count2)/(count1*1.0)
p5 = (count4*1.0)/count3
p6 = (count3-count4)/(count3*1.0)
p7 = count1 + count3 + count5
p8 = (count6*1.0)/count5
p9 = (count5 - count6)/(count5*1.0)
p1 = -1*((count1*1.0)/p7)*(p3*math.log(p3,2) + p4*math.log(p4,2)) + -1*((count3*1.0)/p7)*(p5*math.log(p5,2) + p6*math.log(p6,2)) + -1*((count5*1.0)/p7)*(p8*math.log(p8,2) + p9*math.log(p9,2))
inf_gain = H - p1
print (inf_gain)
print ("###########SALES#########################################")
'sales', 'accounting', 'technical', 'management', 'IT', 'product_mng', 'marketing', 'RandD', 'support', 'hr'
col1 = result[:,7]
count1 = 0
count2 = 0
count3 = 0
count4 = 0
count5 = 0
count6 = 0
count7 = 0
count8 = 0
count9 = 0
count10 = 0
count11 = 0
count12 = 0
count13 = 0
count14 = 0
count15 = 0
count16 = 0
count17 = 0
count18 = 0
count19 = 0
count20 = 0
for i in range(num_rows):
    if (col1[i]) == answer7[0]:
        count1 +=1
    if (col1[i]) == answer7[0] and label[i] == '0':
        count2 +=1
    if (col1[i]) == answer7[1]:
        count3 +=1
    if (col1[i]) == answer7[1] and label[i] == '0':
        count4 +=1
    if (col1[i]) == answer7[2]:
        count5 +=1
    if (col1[i]) == answer7[2] and label[i] == '0':
        count6 +=1
    if (col1[i]) == answer7[3]:
        count7 +=1
    if (col1[i]) == answer7[3] and label[i] == '0':
        count8 +=1
    if (col1[i]) == answer7[4]:
        count9 +=1
    if (col1[i]) == answer7[4] and label[i] == '0':
        count10 +=1
    if (col1[i]) == answer7[5]:
        count11 +=1
    if (col1[i]) == answer7[5] and label[i] == '0':
        count12 +=1
    if (col1[i]) == answer7[6]:
        count13 +=1
    if (col1[i]) == answer7[6] and label[i] == '0':
        count14 +=1
    if (col1[i]) == answer7[7]:
        count15 +=1
    if (col1[i]) == answer7[7] and label[i] == '0':
        count16 +=1
    if (col1[i]) == answer7[8]:
        count17 +=1
    if (col1[i]) == answer7[8] and label[i] == '0':
        count18 +=1
    if (col1[i]) == answer7[9]:
        count19 +=1
    if (col1[i]) == answer7[9] and label[i] == '0':
        count20 +=1
p3 = (count2*1.0)/count1
p4 = (count1-count2)/(count1*1.0)
p5 = (count4*1.0)/count3
p6 = (count3-count4)/(count3*1.0)
p8 = (count6*1.0)/count5
p9 = (count5 - count6)/(count5*1.0)
p10 = (count8*1.0)/count7
p11 = (count7 - count8*1.0)/count7
p12 = (count10*1.0)/count9
p13 = (count9 - count10*1.0)/count9
p14 = (count12*1.0)/count11
p15 = (count11 - count12*1.0)/count11
p16 = (count14*1.0)/count13
p17 = (count13 - count14*1.0)/count13
p18 = (count16*1.0)/count15
p19 = (count15 - count16*1.0)/count15
p20 = (count18*1.0)/count17
p21 = (count17 - count18*1.0)/count17
p22 = (count20*1.0)/count19
p23 = (count19 - count20*1.0)/count19
p7 = count1 + count3 + count5 + count7 + count9 + count11 + count13 + count15 + count17 + count19
p1 = -1*((count1*1.0)/p7)*(p3*math.log(p3,2) + p4*math.log(p4,2)) + -1*((count3*1.0)/p7)*(p5*math.log(p5,2) + p6*math.log(p6,2)) + -1*((count5*1.0)/p7)*(p8*math.log(p8,2) + p9*math.log(p9,2)) + -1*((count7*1.0)/p7)*(p10*math.log(p10,2) + p11*math.log(p11,2)) + -1*((count9*1.0)/p7)*(p12*math.log(p12,2) + p13*math.log(p13,2)) + -1*((count11*1.0)/p7)*(p14*math.log(p14,2) + p15*math.log(p15,2)) + -1*((count13*1.0)/p7)*(p16*math.log(p16,2) + p17*math.log(p17,2)) + -1*((count15*1.0)/p7)*(p18*math.log(p18,2) + p19*math.log(p19,2)) + -1*((count17*1.0)/p7)*(p20*math.log(p20,2) + p21*math.log(p21,2)) + -1*((count19*1.0)/p7)*(p22*math.log(p22,2) + p23*math.log(p23,2)) 
inf_gain = H - p1
print (inf_gain)
