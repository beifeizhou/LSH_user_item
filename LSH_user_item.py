import numpy as np
import pandas as pd 
import os
import sys

inputpath = '/Users/nali/Beifei/ximalaya2015/code_ximalaya/code_LSH/input/newfeatures_leo_20160108/'
javapath = '/Users/nali/Beifei/ximalaya2015/code_ximalaya/code_LSH/TarsosLSH/TarsosLSH_test/build/TarsosLSH-0.7.jar'

com1 = 'rm result.txt'
com2 = 'rm *.bin'
com3 = "java -jar "+javapath+" -d itemfeature.txt -q userfeature.txt -n 500 -f cos > result.txt"
com4 = 'grep ";" result.txt > LSHresult.txt'

os.system(com1)
os.system(com2)
os.system(com3)
os.system(com4)

a = pd.read_csv('LSHresult.txt', header = None, sep=';')
a = a[range(a.shape[1]-1)]
a = a.values.tolist()
a = map(lambda x: (x[0][:-1][1:], map(lambda xx: xx[:-1][1:], x[1:])), a) 

queries = pd.read_csv(inputpath+'userfeature.txt', header = None, sep = ' ')
item_vec = pd.read_csv(inputpath+'itemfeature.txt', header = None, sep = ' ')

user_vec_dict = dict(map(lambda x: (x[0][:-1][1:], x[1:]), queries.values))
item_vec_dict = dict(map(lambda x: (x[0][:-1][1:], x[1:]), item_vec.values))

def rearrange(x1, x2):
    user_v = user_vec_dict[x1]
    item_v = np.array(map(lambda x: item_vec_dict[x], x2))
    s = np.dot(user_v, item_v.T)
    item_s = zip(x2, s * -1)
    item_final = sorted(item_s, key = lambda x: x[1])
    return (x1, map(lambda x: x[0], item_final)[:50])

a = map(lambda x: rearrange(x[0], x[1]), a)
output50 = open('lsh_500_50.txt', 'w')
for i in range(len(a)):
    b = a[i]
    each = b[0]+','+','.join(b[1])+'\n'
    output50.write(each)
output50.close()

LSHresults_dict = dict(a)

users = map(lambda x: x[:-1][1:], queries[0])
open('users.txt', 'w').write('\n'.join(users))

queries = queries[range(1,21)]
items = map(lambda x: x[:-1][1:], item_vec[0])
item_vec = item_vec[range(1,21)]
dict_I = dict(zip(range(len(items)), items))

score = np.dot(queries, item_vec.T)
Order = (-1*score).argsort()
top100 = Order[:,range(50)]
def map_back(x):
    items = map(lambda xx: str(dict_I[xx]), x)
    return items

top100 = np.apply_along_axis(map_back, 1, top100)
results_dict = dict(zip(users, top100.tolist()))

output = open('intersection.txt', 'w')

for i in range(len(users)):
    lsh = LSHresults_dict[users[i]]
    score = results_dict[users[i]]
    inter = set(lsh).intersection(set(score))
    output.write(users[i]+','+str(len(inter))+'\n')

