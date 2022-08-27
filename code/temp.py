import numpy as np
import pickle
import sparse
import time
import matplotlib.pyplot as plt

path = '../tables/conn_table/cortical_v2/conn.pickle'

with open(path, 'rb') as f:
    data = pickle.load(f)

matrix = np.ones((1000, 1000))

time1 = time.time()
sum_elem = 0
for i in range(1000):
    for j in range(1000):
        sum_elem += matrix[i][j]
time2 = time.time()
print("%.6f seconds consumed for 1e6 indexing" % (time2 - time1))


time3 = time.time()
sum_elem = 0
for j in range(data.shape[0]):
    sum_elem += data[0][j]

time4 = time.time()
print("%.6f seconds consumed for 1e6 indexing" % ((time4 - time3) / data.shape[0] * 1e6))
