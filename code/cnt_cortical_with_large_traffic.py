import numpy as np
import pickle
import time

path = '../tables/traffic_table/'
file_name = "traffic_base_cortical_map_10000_split_4_cortical_v2.pkl"

map_path = "../tables/map_table/map_10000_split_4_cortical_v2_without_invalid_idx.pkl"

N = 10000
n = 171508

print(file_name)

time1 = time.time()
traffic_base_cortical_path = path + file_name

with open(traffic_base_cortical_path, 'rb') as f:
    traffic_base_cortical = pickle.load(f)
print(traffic_base_cortical_path + ' loaded.')
time2 = time.time()
print("%.2f seconds consumed." % (time2 - time1))

with open(map_path, 'rb') as f:
    map_table = pickle.load(f)

print("map table loaded.")


# 计算每个population向外发送的流量大小
time1 = time.time()

# 没有split
# traffic_per_cortical = np.zeros(n)
# for src_idx in range(N):
#     for dst_idx in range(N):
#         if src_idx != dst_idx:
#             for i in range(len(map_table[str(src_idx)])):
#                 p_idx = map_table[str(src_idx)][i]
#                 traffic_per_cortical[p_idx] += traffic_base_cortical[dst_idx][src_idx][i]
#     if src_idx % 100 == 0:
#         print(src_idx)

# split的population流量统计
traffic_gpu_population = list()
for src_idx in range(N):
    traffic_for_1_gpu = list()
    keys = map_table[str(src_idx)].keys()
    for i in range(len(keys)):
        sum_traffic = 0
        for dst_idx in range(N):
            if src_idx // 4 != dst_idx // 4:
                sum_traffic += traffic_base_cortical[src_idx][i]
        if src_idx % 1000 == 0:
            print(src_idx)
    traffic_gpu_population.append(traffic_for_1_gpu)
time2 = time.time()
print('%.2f' % (time2 - time1))

np.save('traffic_per_population.npy', traffic_per_cortical)

max_times = 10

lst = np.zeros(max_times + 1, dtype=int)
average = np.sum(traffic_per_cortical) / 10000
for i in range(n):
    for j in range(1, max_times + 1, 1):
        if traffic_per_cortical[i] >= j * average:
            lst[j] += 1

for i in range(1, max_times + 1, 1):
    print("%d: %d" % (i, lst[i]))

print()
