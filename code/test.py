"""
Check if the connection number of different process is the same, thus proving the correctness of route_table
"""

import pickle
import time
import numpy as np


N = 8000

# 按照dti连接表算出来的连接数
with open('../tables/route_table/route_v1_map_8000_v1/new_connection_2.pkl', 'rb') as f:
    updated_route_in = pickle.load(f)

updated_route_out = [[] for _ in range(N)]
for dst in range(N):
    for src in updated_route_in[dst]:
        updated_route_out[src].append(dst)

connection_num_per_dcu_new_in = np.zeros(N)
connection_num_per_dcu_new_out = np.zeros(N)
for i in range(8000):
    connection_num_per_dcu_new_in[i] = len(updated_route_in[i])
    connection_num_per_dcu_new_out[i] = len(updated_route_out[i])
print('New connection_in:', np.min(connection_num_per_dcu_new_in),
      np.max(connection_num_per_dcu_new_in),
      np.average(connection_num_per_dcu_new_in))
print('New connection_out:', np.min(connection_num_per_dcu_new_out),
      np.max(connection_num_per_dcu_new_out),
      np.average(connection_num_per_dcu_new_out))

check_result = True
for i in range(N):
    if connection_check_in[i] > connection_num_per_dcu_new_in[i]:
        check_result = False
        break
print(check_result)

print('hey')
