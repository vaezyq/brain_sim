import numpy as np
import time


def cal_genuine_link_number_base_dense_route_table(number_of_group=20, dcu_per_group=40):
    N = 800
    binary_connection_table_base_dcu = np.load("traffic_table/traffic_table_base_dcu_map_1600_v2_cortical_v2.npy",
                                               allow_pickle=True)
    binary_connection_table_base_dcu = np.array(binary_connection_table_base_dcu, dtype=object)
    binary_connection_table_base_dcu_bool = np.array(binary_connection_table_base_dcu, dtype=bool)
    # binary_connection_table_base_dcu_bool = np.ones((800, 800), dtype=int)
    route_table = np.load('route_table/route_default_2dim_20_40.npy')

    inside_group_link_num = np.zeros(N, dtype=int)
    between_group_link_num = np.zeros(N, dtype=int)

    time1 = time.time()
    for i in range(N):
        # 先统计各个节点负责的dcu有哪些
        masters_and_slaves = dict()
        for j in range(N):
            temp = route_table[i][j]
            if temp not in masters_and_slaves:
                masters_and_slaves[temp] = list()
            masters_and_slaves[temp].append(j)

        for j in range(N):
            if j not in masters_and_slaves and j in masters_and_slaves[i]:  # 组间
                if binary_connection_table_base_dcu_bool[j][i] == 1:
                    inside_group_link_num[i] += 1
            elif j in masters_and_slaves:  # 组内
                for idx in masters_and_slaves[j]:
                    if binary_connection_table_base_dcu_bool[idx][i] == 1:
                        between_group_link_num[i] += 1
                        break
        if i % 1000 == 0:
            print(i)
    time2 = time.time()

    link_num = inside_group_link_num + between_group_link_num
    np.save('inside_group_%d_%d.npy' % (number_of_group, dcu_per_group), inside_group_link_num)
    np.save('between_group_%d_%d.npy' % (number_of_group, dcu_per_group), between_group_link_num)
    np.save('link_num_%d_%d.npy' % (number_of_group, dcu_per_group), link_num)
    print('########### %d * %d ###########' % (number_of_group, dcu_per_group))
    print('inside group:', np.max(inside_group_link_num), np.min(inside_group_link_num),
          np.average(inside_group_link_num))
    print('between group:', np.max(between_group_link_num), np.min(between_group_link_num),
          np.average(between_group_link_num))
    print('sum:', np.max(link_num), np.min(link_num), np.average(link_num))
    print('%.2f seconds consumed.' % (time2 - time1))


def get_route_dict_out(route_dict_table, out_idx, in_idx_list):
    route_dict_out_idx = {}
    for idx in in_idx_list:
        if idx == out_idx:
            continue
        else:
            # 直接发送
            if route_dict_table[out_idx][idx] == out_idx and not (idx in route_dict_out_idx):
                route_dict_out_idx[int(idx)] = {idx}
            else:
                if not (route_dict_table[out_idx][idx] in route_dict_out_idx):
                    route_dict_out_idx[int(route_dict_table[out_idx][idx])] = {
                        route_dict_table[out_idx][idx]}
                route_dict_out_idx[int(route_dict_table[out_idx][idx])].add(idx)

    return route_dict_out_idx


def cal_genuine_link_number_base_dense_route_table_random_dim(N, connection_table_path, route_table_path,
                                                              dimensionSize):
    binary_connection_table_base_dcu = np.load(connection_table_path, allow_pickle=True)
    binary_connection_table_base_dcu = np.array(binary_connection_table_base_dcu, dtype=object)
    binary_connection_table_base_dcu_bool = np.array(binary_connection_table_base_dcu, dtype=bool)
    route_table = np.load(route_table_path)

    inside_group_link_num = np.zeros((N, N), dtype=int)
    between_group_link_num = np.zeros((N, N), dtype=int)

    time1 = time.time()
    for idx in range(N):
        route_dict_out_idx = get_route_dict_out(route_table, idx, range(N))
        for in_idx, in_idx_list in route_dict_out_idx.items():
            if len(in_idx_list) == 1:
                if binary_connection_table_base_dcu_bool[idx][in_idx] == 1:
                    inside_group_link_num[idx][in_idx] = 1
            else:
                for i in in_idx_list:
                    if binary_connection_table_base_dcu_bool[idx][i] == 1 and idx != in_idx:
                        between_group_link_num[idx][in_idx] = 1

    inside_group_link_num = np.sum(inside_group_link_num, axis=0)
    between_group_link_num = np.sum(between_group_link_num, axis=0)

    # print("--------")
    # print(between_group_link_num)
    # print(between_group_link_num)
    time2 = time.time()

    link_num = inside_group_link_num + between_group_link_num
    str1 = ''
    str2 = ''
    for i in dimensionSize:
        str1 += '_'
        str1 += str(i)
        str2 += '*'
        str2 += str(i)

    np.save('inside_group_%s.npy' % str1, inside_group_link_num)
    np.save('between_group_%s.npy' % str1, between_group_link_num)
    np.save('link_num_%s.npy' % str1, link_num)
    print('########### %s ###########' % str2[1:])
    print('inside group:', np.max(inside_group_link_num), np.min(inside_group_link_num),
          np.average(inside_group_link_num))
    print('between group:', np.max(between_group_link_num), np.min(between_group_link_num),
          np.average(between_group_link_num))
    print('sum:', np.max(link_num), np.min(link_num), np.average(link_num))
    print('%.2f seconds consumed.' % (time2 - time1))


def calculate_one_dim_conn(connection_table_path):
    binary_connection_table_base_dcu = np.load(connection_table_path, allow_pickle=True)
    binary_connection_table_base_dcu = np.array(binary_connection_table_base_dcu, dtype=object)
    binary_connection_table_base_dcu_bool = np.array(binary_connection_table_base_dcu, dtype=bool)
    link_num = np.sum(binary_connection_table_base_dcu_bool, axis=0)
    print('sum:', np.max(link_num), np.min(link_num), np.average(link_num))
