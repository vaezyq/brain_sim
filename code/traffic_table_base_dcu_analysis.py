import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import scipy.sparse
from scipy.sparse import coo_matrix

number_of_groups = 17280
n_dcu_per_group = 1

traffic_table_base_dcu = np.load('../tables/traffic_table/traffic_table_base_dcu_before.npy')
binary_traffic_table_base_dcu = np.array(traffic_table_base_dcu, dtype=bool)
# binary_conn_table_after_grouping = np.load('../tables/traffic_table/binary_conn_table_after_grouping')
N = traffic_table_base_dcu.shape[0]

print(np.count_nonzero(binary_traffic_table_base_dcu) / (number_of_groups**2))

temp = scipy.sparse.random(10, 10, density=0.5)

fig = plt.figure(figsize=(10, 10), dpi=500)
ax = fig.add_subplot(111)
ax.matshow(binary_traffic_table_base_dcu, cmap='binary')
ax.set_xticks([i for i in range(0, 17280+1920, 1920)])
ax.set_yticks([i for i in range(0, 17280+1920, 1920)])
plt.savefig('conn_base_dcu_before.png')
plt.show()

# binary_conn_table_after_grouping = np.zeros((number_of_groups, number_of_groups))

# print('Begin Calculating...')
# for row in range(number_of_groups):
#     for col in range(number_of_groups):
#         for i in range(n_dcu_per_group):
#             temp = False
#             row_idx = row * n_dcu_per_group + i
#             for j in range(n_dcu_per_group):
#                 col_idx = col * n_dcu_per_group + j
#                 if binary_traffic_table_base_dcu[row_idx][col_idx] == 1:
#                     binary_conn_table_after_grouping[row][col] = True
#                     temp = True
#                     break
#             if temp:
#                 break
#     print(row)
#
# print(np.count_nonzero(binary_conn_table_after_grouping) / (number_of_groups**2))

# np.save('../tables/traffic_table/binary_conn_table_after_grouping', binary_conn_table_after_grouping)
