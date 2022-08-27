"""
这个类用于分析生成的体素->dcu映射表
"""
import os.path

import numpy as np
import time
import matplotlib.pyplot as plt
import torch
from generate_map import GenerateMap
from parallelism import Parallelism
import copy
import os


# import torch


class MapAnalysis(GenerateMap):
    def __init__(self):
        super().__init__()

        self.map_table = self.read_map_pkl(self.map_table_path)

        if self.conn_version[0:5] == 'voxel':
            self.map_table_without_invalid_idx = self.map_table
        else:
            self.map_table_without_invalid_idx = self.read_map_pkl(self.map_table_without_invalid_idx_path)
        self.traffic_voxel_to_voxel = np.array([])

    def cal_probability_per_dcu(self):
        probability_per_dcu = np.zeros(self.N)
        out_size_per_voxel = np.zeros(self.n)

        for i in range(self.n):
            out_size_per_voxel[i] = np.sum(self.conn[:, i])

        for i in range(self.N):
            for voxel_idx in self.map_table[i]:
                probability_per_dcu[i] += out_size_per_voxel[voxel_idx]

        print(np.max(probability_per_dcu), np.average(probability_per_dcu))

        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 7), dpi=100)
        plt.title('probability_per_dcu: max = %.4f, average = %.4f, max/average = %.4f'
                  % (np.max(probability_per_dcu), np.average(probability_per_dcu),
                     np.max(probability_per_dcu / np.average(probability_per_dcu))))
        plt.plot(probability_per_dcu)
        plt.show()

        traffic_base_dcu = np.load('traffic_table_base_dcu_map_2000_v2.npy')
        sum_out = np.zeros(self.N)
        sum_in = np.zeros(self.N)

        for i in range(self.N):
            sum_out[i] = np.sum(traffic_base_dcu[:, i])
            sum_in[i] = np.sum(traffic_base_dcu[i, :])

        plt.figure(figsize=(12, 7), dpi=100)
        plt.title('sum_out: max = %.4f, average = %.4f, max/average = %.4f'
                  % (np.max(sum_out), np.average(sum_out), np.max(sum_out / np.average(sum_out))))
        plt.plot(sum_out)
        plt.show()

        plt.figure(figsize=(12, 7), dpi=100)
        plt.title('sum_in: max = %.4f, average = %.4f, max/average = %.4f'
                  % (np.max(sum_in), np.average(sum_in), np.max(sum_in / np.average(sum_in))))
        plt.plot(sum_in)
        plt.show()

        print('hey')

    # 计算划分后，交换机之间的流量
    def cal_flow_under_switch(self):
        map_table = self.map_table

        dcus_per_switch = list()  # 每个交换机下的dcu
        for i in range(self.number_of_switches):
            dcus_per_switch.append([])

        idx = 0
        for i in range(self.number_of_switches):
            for j in range(self.gpu_per_switch):
                dcus_per_switch[i].append(map_table[idx])
                idx += 1

        flow_between_switches = list()  # 交换机两两之间的流量
        for i in range(self.number_of_switches):
            flow_between_switches.append([0] * self.number_of_switches)

        print('Begin calculation...')
        time_start = time.time()
        for i in range(self.number_of_switches):
            for j in range(self.number_of_switches):
                for dcu_out in dcus_per_switch[i]:
                    for voxel_out in dcu_out:
                        for dcu_in in dcus_per_switch[j]:
                            for voxel_in in dcu_in:
                                flow_between_switches[i][j] += self.conn[voxel_out][voxel_in] * self.size[voxel_out]
            # self.show_progress(i, self.number_of_switches, time_start)
        time_end = time.time()
        print('Calculation of the flow between switches complete, %.2fs consumed.' % (time_end - time_start))
        print('hey')

    # 找个数大于1的最小平均值的索引
    @staticmethod
    def find_min_max_idx(initial, average):
        idx_max = int(np.argmax(average))

        idx_all = np.argsort(average)
        temp = 0
        while initial[temp] == 1:
            temp += 1
        idx_min = idx_all[temp]

        return idx_max, idx_min

    # 计算每个dcu平均负责的流量
    def cal_average(self, switch_idx, single_flow, initial):
        average_flow = list()
        for j in range(self.number_of_switches):
            if j < switch_idx:
                average_flow.append(single_flow[j] / initial[j])
            elif j > switch_idx:
                average_flow.append(single_flow[j] / initial[j - 1])
        average_flow = np.array(average_flow)
        return average_flow

    # 根据交换机之间的流量，得出对于每个交换机，向其他交换机发送消息时，要由哪些dcu转发
    def cal_forwarding_table_old(self, iter_times=60):
        flow_between_switches = np.loadtxt('flow_between_switches.txt')  # 交换机两两之间的流量
        forward_table = list()  # 最后输出的dcu负责交换机的情况

        # 初始的负责每个交换机的dcu数量
        initial_num = {3: 16, 4: 8}
        single_forward_table = list()
        for key in initial_num:
            for i in range(initial_num[key]):
                single_forward_table.append(key)

        # 对于每个交换机，迭代若干次以生成初始转发表
        for i in range(self.number_of_switches):
            single_flow = flow_between_switches[i]

            # 迭代若干次，更新initial_num
            initial = single_forward_table.copy()
            average_flow = self.cal_average(i, single_flow, initial)
            for j in range(iter_times):
                idx_max, idx_min = self.find_min_max_idx(initial, average_flow)
                initial[idx_max] += 1
                initial[idx_min] -= 1
                # 更新average_flow
                average_flow = self.cal_average(i, single_flow, initial)

            initial.insert(i, 0)

            ret = list()  # 将dcu的数量转化为编号
            cnt = i * self.gpu_per_switch  # 注意
            for j in range(self.number_of_switches):
                ret.append([])
                for k in range(initial[j]):
                    ret[j].append(cnt)
                    cnt += 1

            forward_table.append(ret)

        return forward_table

    # 每n_dcu_per_group个dcu分为1个group，计算每个group到其他所有dcu的流量
    def cal_flow_between_group_and_dcu(self):
        map_table = self.map_table

        dcus_per_group = list()  # 每个group下包含的dcu的绝对编号
        for i in range(self.number_of_groups):
            dcus_per_group.append([])
            for j in range(self.n_gpu_per_group):
                dcus_per_group[i].append(self.n_gpu_per_group * i + j)

        flow_between_group_and_dcu = list()  # 所有group到其他所有dcu的流量
        for i in range(self.number_of_groups):
            flow_between_group_and_dcu.append([0] * self.N)

        print('Begin calculation...')
        time_start = time.time()
        for group_out_idx in range(self.number_of_groups):
            for dcu_out in dcus_per_group[group_out_idx]:
                for voxel_out in map_table[dcu_out]:
                    for dcu_in in range(self.N):
                        for voxel_in in map_table[dcu_in]:
                            flow_between_group_and_dcu[group_out_idx][dcu_in] += \
                                self.conn[voxel_out][voxel_in] * self.size[voxel_out]

                # self.show_progress(dcu_out, self.N, time_start)
        time_end = time.time()
        # np.savetxt('C:/all/WOW/brain/partition_and_route/flow_table/flow_2000_base_node.txt',
        #            flow_between_group_and_dcu)
        print('Calculation of the flow between group and dcu complete, %.2fs consumed.' % (time_end - time_start))
        print('hey')

    # 根据group之间的流量，得出对于每个group，向其他dcu发送消息时，要由group内的哪些dcu转发
    def generate_forwarding_table(self, iter_times=300, max_link=147):
        flow_table_path = 'C:/all/WOW/brain/partition_and_route/flow_table/flow_2000dcu_40.txt'
        flow_between_group_and_dcu = np.loadtxt(flow_table_path)  # group到其他dcu的流量

        forward_table = list()  # 最后输出的dcu负责交换机的情况
        flow_table = list()

        # 对于每个group，迭代若干次以生成初始转发表
        for i in range(self.number_of_groups):
            single_flow_table = flow_between_group_and_dcu[i]

            single_forward_table = list()  # 生成初始解
            idx = 0
            for j in range(self.n_gpu_per_group):
                single_forward_table.append([])
                for k in range(self.number_of_groups - 1):
                    while idx // self.n_gpu_per_group == i:
                        idx += 1
                    single_forward_table[j].append(idx)
                    idx += 1

            flow_each_dcu = np.zeros(self.n_gpu_per_group)  # 计算流量
            for j in range(self.n_gpu_per_group):
                for dcu_in in single_forward_table[j]:
                    flow_each_dcu[j] += single_flow_table[dcu_in]

            # origin = flow_each_dcu.copy()
            # plt.ion()
            # while np.max(flow_each_dcu) > 0.00025:
            for j in range(iter_times):
                max_idx = int(np.argmax(flow_each_dcu))
                min_idx = int(np.argmin(flow_each_dcu))

                # 从负责转发的流量最大的dcu中，随机选择一个其负责转发的dcu，放入流量最小的dcu的负责转发列表中去
                random_idx = np.random.randint(0, len(single_forward_table[max_idx]))
                dcu_to_move = single_forward_table[max_idx][random_idx]
                single_forward_table[max_idx] = list(np.delete(single_forward_table[max_idx], random_idx))
                single_forward_table[min_idx] = list(np.append(single_forward_table[min_idx], dcu_to_move))
                flow_each_dcu[max_idx] -= single_flow_table[dcu_to_move]
                flow_each_dcu[min_idx] += single_flow_table[dcu_to_move]

                # 如果连接数超过要求，从超过要求的dcu中拿出一个流量小的dcu
                if len(single_forward_table[min_idx]) >= max_link - self.n_gpu_per_group + 2:
                    flows = list()
                    for dcu_in in range(len(single_forward_table[min_idx])):
                        flows.append(single_flow_table[dcu_in])
                    sort_idx = np.argsort(flows)
                    rand_idx = np.random.randint(0, 5)
                    temp = single_forward_table[min_idx][sort_idx[rand_idx]]
                    single_forward_table[min_idx] = list(np.delete(single_forward_table[min_idx], sort_idx[rand_idx]))
                    single_forward_table[max_idx] = list(np.append(single_forward_table[max_idx], temp))
                    flow_each_dcu[min_idx] -= single_flow_table[temp]
                    flow_each_dcu[max_idx] -= single_flow_table[temp]

                # plt.clf()
                # plt.title(j)
                # plt.ylim(0.0003, 0.0005)
                # plt.plot(origin, color='blue', linestyle='--', alpha=0.4)
                # plt.plot(flow_each_dcu, color='blue')
                # plt.pause(0.001)
            #
            # max_len = 0
            # for j in range(len(single_forward_table)):
            #     max_len = max(max_len, len(single_forward_table[j]))
            # print('max len = %d' % max_len)

            forward_table.append(single_forward_table)
            flow_table.append(flow_each_dcu)

        flow_array = list()
        for i in range(self.number_of_groups):
            for j in range(self.n_gpu_per_group):
                flow_array.append(flow_table[i][j])
        # np.savetxt('C:/all/WOW/brain/partition_and_route/route_4000_v1/level2_flow_before.txt', flow_array)

        return forward_table

    def cal_flow_between_group_and_node(self):
        flow_table_path = 'C:/all/WOW/brain/partition_and_route/flow_table/flow_2000dcu_40.txt'
        flow_between_group_and_dcu = np.loadtxt(flow_table_path)  # group到其他dcu的流量

        flow_between_group_and_node = np.zeros((self.number_of_groups, self.number_of_nodes))

        for i in range(self.number_of_groups):
            for j in range(self.number_of_nodes):
                flow_between_group_and_node[i][j] = np.sum(
                    flow_between_group_and_dcu[2 * i: 2 * i + 2, 4 * j: 4 * j + 4])

        np.save('C:/all/WOW/brain/partition_and_route/flow_table/flow_between_25group_and_500node.npy',
                flow_between_group_and_node)

        return

    def generate_forwarding_table_base_node(self, iter_times=30, max_link=51):
        flow_table_path = 'C:/all/WOW/brain/partition_and_route/flow_table/flow_between_25group_and_500node.txt'
        flow_between_group_and_node = np.loadtxt(flow_table_path)  # group到其他node的流量

        forward_table = list()  # 最后输出的dcu负责交换机的情况
        flow_table = list()

        # 对于每个group，迭代若干次以生成初始转发表
        for i in range(self.number_of_groups):
            single_flow_table = flow_between_group_and_node[i]

            # 生成初始解
            single_forward_table = list()
            idx = 0
            for j in range(self.n_node_per_group):
                single_forward_table.append([])
                for k in range(self.number_of_groups - 1):
                    while idx // self.n_node_per_group == i:
                        idx += 1
                    single_forward_table[j].append(idx)
                    idx += 1

            # 计算流量
            flow_each_node = np.zeros(self.n_node_per_group)
            for j in range(self.n_node_per_group):
                for node_in in single_forward_table[j]:
                    flow_each_node[j] += single_flow_table[node_in]

            # origin = flow_each_node.copy()
            # plt.ion()

            # 迭代降低每个node负责转发的流量
            # for j in range(iter_times):  # 指定迭代系数的循环条件
            cnt_iter = 0
            while np.max(flow_each_node) > 0.001837 or cnt_iter < iter_times:  # 限制流量的循环条件
                cnt_iter += 1
                max_idx = int(np.argmax(flow_each_node))
                min_idx = int(np.argmin(flow_each_node))

                # 从负责转发的流量最大的node中，随机选择一个其负责转发的node，放入流量最小的node负责转发的列表中去
                random_idx = np.random.randint(0, len(single_forward_table[max_idx]))
                node_to_move = single_forward_table[max_idx][random_idx]
                single_forward_table[max_idx] = list(np.delete(single_forward_table[max_idx], random_idx))
                single_forward_table[min_idx] = list(np.append(single_forward_table[min_idx], node_to_move))
                flow_each_node[max_idx] -= single_flow_table[node_to_move]
                flow_each_node[min_idx] += single_flow_table[node_to_move]

                # 如果连接数超过要求，从超过要求的node中随机拿出一个流量第0-2小的dcu
                if len(single_forward_table[min_idx]) >= max_link - self.n_node_per_group + 2:
                    flows = list()
                    for node_in in range(len(single_forward_table[min_idx])):
                        flows.append(single_flow_table[node_in])
                    sort_idx = np.argsort(flows)
                    rand_idx = np.random.randint(0, 5)
                    temp = single_forward_table[min_idx][sort_idx[rand_idx]]
                    single_forward_table[min_idx] = list(np.delete(single_forward_table[min_idx], sort_idx[rand_idx]))
                    single_forward_table[max_idx] = list(np.append(single_forward_table[max_idx], temp))
                    flow_each_node[min_idx] -= single_flow_table[temp]
                    flow_each_node[max_idx] -= single_flow_table[temp]

                # plt.clf()
                # # plt.title(j)
                # plt.ylim(0.0010, 0.0016)
                # plt.plot(origin, color='blue', linestyle='--', alpha=0.4)
                # plt.plot(flow_each_node, color='blue')
                # plt.pause(0.001)
            # max_len = 0
            # for j in range(len(single_forward_table)):
            #     max_len = max(max_len, len(single_forward_table[j]))
            # print('max len = %d' % max_len)

            forward_table.append(single_forward_table)
            flow_table.append(flow_each_node)

        # 计算每个node的2级路由流量
        flow_array = list()
        for i in range(self.number_of_groups):
            for j in range(self.n_node_per_group):
                flow_array.append(flow_table[i][j])
        np.save(self.route_path + 'level2_route_traffic.npy', flow_array)

        # 画图看流量变化
        plt.figure(figsize=(10, 6))
        plt.title('max=%.5f, min=%.5f, average=%.5f' %
                  (np.max(flow_array), np.min(flow_array), np.average(flow_array)))
        plt.xlabel('node')
        plt.ylabel('traffic')
        plt.ylim(0.0008, 0.0024)
        plt.plot(flow_array)
        plt.savefig(self.route_path + 'level2_route_traffic.png', dpi=200)
        plt.show()

        return forward_table

    def cal_flow_between_group_and_dcu_unpack_inside_group(self):
        map_table = self.map_table

        dcus_per_group = list()  # 每个group下包含的dcu的绝对编号
        for i in range(self.number_of_groups):
            dcus_per_group.append([])
            for j in range(self.n_gpu_per_group):
                dcus_per_group[i].append(self.number_of_groups * j + i)

        flow_between_group_and_dcu = list()  # 所有group到其他所有dcu的流量
        for i in range(self.number_of_groups):
            flow_between_group_and_dcu.append([0] * self.N)

        print('Begin calculation...')
        time_start = time.time()
        cnt = 0
        for group_out_idx in range(self.number_of_groups):
            for dcu_out in dcus_per_group[group_out_idx]:
                cnt += 1
                for voxel_out in map_table[dcu_out]:
                    for dcu_in in range(self.N):
                        for voxel_in in map_table[dcu_in]:
                            flow_between_group_and_dcu[group_out_idx][dcu_in] += \
                                self.conn[voxel_out][voxel_in] * self.size[voxel_out]

                # self.show_progress(cnt, self.N, time_start)
        end_time = time.time()
        print("Calculation of flow between group and dcu complete. %.2fs consumed." % (end_time - time_start))
        time_end = time.time()
        np.savetxt('C:/all/WOW/brain/partition_and_route/flow_table/flow_2000dcu_40_unpack_inside_group.txt',
                   flow_between_group_and_dcu, dtype=int)
        print('Calculation of the flow between group and dcu complete, %.2fs consumed.' % (time_end - time_start))
        print('hey')

    def generate_forwarding_table_unpack_inside_group(self, iter_times=80, max_link=92):
        flow_table_path = 'C:/all/WOW/brain/partition_and_route/flow_table/flow_2000dcu_40_unpack_inside_group.txt'
        flow_between_group_and_dcu = np.loadtxt(flow_table_path)  # group到其他dcu的流量

        forward_table = list()  # 最后输出的group负责dcu的情况
        flow_table = list()

        # 对于每个group，迭代若干次以生成转发表
        for i in range(self.number_of_groups):
            single_flow_table = flow_between_group_and_dcu[i]

            single_forward_table = list()  # 生成初始解
            idx = 0
            for j in range(self.n_gpu_per_group):
                single_forward_table.append([])
                for k in range(self.number_of_groups):
                    if idx != self.number_of_groups * j + i:
                        single_forward_table[j].append(idx)
                    idx += 1

            flow_each_dcu = np.zeros(self.n_gpu_per_group)  # 计算流量
            for j in range(self.n_gpu_per_group):
                for dcu_in in single_forward_table[j]:
                    flow_each_dcu[j] += single_flow_table[dcu_in]

            # origin = flow_each_dcu.copy()
            # plt.ion()
            # while np.max(flow_each_dcu) > 0.00025:

            for j in range(iter_times):
                max_idx = int(np.argmax(flow_each_dcu))
                min_idx = int(np.argmin(flow_each_dcu))

                # 从负责转发的流量最大的dcu中，随机选择一个其负责转发的dcu，放入流量最小的dcu的负责转发列表中去
                random_idx = np.random.randint(0, len(single_forward_table[max_idx]))
                dcu_to_move = single_forward_table[max_idx][random_idx]
                single_forward_table[max_idx] = list(np.delete(single_forward_table[max_idx], random_idx))
                single_forward_table[min_idx] = list(np.append(single_forward_table[min_idx], dcu_to_move))
                flow_each_dcu[max_idx] -= single_flow_table[dcu_to_move]
                flow_each_dcu[min_idx] += single_flow_table[dcu_to_move]

                # 如果连接数超过要求，从超过要求的dcu中拿出一个流量小的dcu
                if len(single_forward_table[min_idx]) >= max_link - self.n_gpu_per_group + 2:
                    flows = list()
                    for dcu_in in range(len(single_forward_table[min_idx])):
                        flows.append(single_flow_table[dcu_in])
                    sort_idx = np.argsort(flows)
                    rand_idx = np.random.randint(0, 5)
                    temp = single_forward_table[min_idx][sort_idx[rand_idx]]
                    single_forward_table[min_idx] = list(
                        np.delete(single_forward_table[min_idx], sort_idx[rand_idx]))
                    single_forward_table[max_idx] = list(np.append(single_forward_table[max_idx], temp))
                    flow_each_dcu[min_idx] -= single_flow_table[temp]
                    flow_each_dcu[max_idx] -= single_flow_table[temp]

                # plt.clf()
                # plt.title('iter times=%d, max=%.6f, average=%.6f' %
                #           (j, np.max(flow_each_dcu), np.average(flow_each_dcu)))
                # plt.ylim(0.0003, 0.0008)
                # plt.plot(origin, color='blue', linestyle='--', alpha=0.4)
                # plt.plot(flow_each_dcu, color='blue')
                # plt.pause(0.001)

            # max_len = 0
            # for j in range(len(single_forward_table)):
            #     max_len = max(max_len, len(single_forward_table[j]))
            # print('max len = %d' % max_len)

            forward_table.append(single_forward_table)
            flow_table.append(flow_each_dcu)

        return forward_table

    def cal_traffic_between_group_and_dcu_unpack_inside_group_with_sampling(self):
        map_table = self.map_table

        dcus_per_group = list()  # 每个group下包含的dcu的绝对编号
        for i in range(self.number_of_groups):
            dcus_per_group.append([])
            for j in range(self.n_gpu_per_group):
                dcus_per_group[i].append(self.number_of_groups * j + i)

        temp = np.load("../tables/traffic_table/out_traffic_voxel_to_voxel_map_v2.npz")
        self.traffic_voxel_to_voxel = temp[temp.files[1]]

        traffic_between_group_and_dcu = np.zeros((self.number_of_groups, self.N))

        for group_idx in range(self.number_of_groups):
            for dcu_in_idx in range(self.N):
                for voxel_in in map_table[dcu_in_idx]:
                    for dcu_out_idx in dcus_per_group[group_idx]:
                        for voxel_out in map_table[dcu_out_idx]:
                            traffic_between_group_and_dcu[group_idx][dcu_in_idx] += \
                                self.traffic_voxel_to_voxel[voxel_out][voxel_in]
            print(group_idx)

        np.save('../tables/traffic_table/group_to_dcu_map_1200_v3.npy', traffic_between_group_and_dcu)
        print('../tables/traffic_table/group_to_dcu_map_1200_v3.npy saved.')

        return traffic_between_group_and_dcu

    def generate_forwarding_table_unpack_inside_group_with_sampling(self, iter_times=80, max_link=92):
        # flow_table_path = '../tables/traffic_table/group_to_dcu_map_1200_v3.npy'
        # flow_between_group_and_dcu = np.load(flow_table_path)
        flow_between_group_and_dcu = self.cal_traffic_between_group_and_dcu_unpack_inside_group_with_sampling()

        forward_table = list()  # 最后输出的group负责dcu的情况
        flow_table = list()

        # 对于每个group，迭代若干次以生成转发表
        for i in range(self.number_of_groups):
            single_flow_table = flow_between_group_and_dcu[i]

            single_forward_table = list()  # 生成初始解
            idx = 0
            for j in range(self.n_gpu_per_group):
                single_forward_table.append([])
                for k in range(self.number_of_groups):
                    if idx != self.number_of_groups * j + i:
                        single_forward_table[j].append(idx)
                    idx += 1

            flow_each_dcu = np.zeros(self.n_gpu_per_group)  # 计算流量
            for j in range(self.n_gpu_per_group):
                for dcu_in in single_forward_table[j]:
                    flow_each_dcu[j] += single_flow_table[dcu_in]

            # origin = flow_each_dcu.copy()
            # plt.ion()
            # for j in range(iter_times):
            while np.max(flow_each_dcu) > 445000:
                max_idx = int(np.argmax(flow_each_dcu))
                min_idx = int(np.argmin(flow_each_dcu))

                # 从负责转发的流量最大的dcu中，随机选择一个其负责转发的dcu，放入流量最小的dcu的负责转发列表中去
                random_idx = np.random.randint(0, len(single_forward_table[max_idx]))
                dcu_to_move = single_forward_table[max_idx][random_idx]
                single_forward_table[max_idx] = list(np.delete(single_forward_table[max_idx], random_idx))
                single_forward_table[min_idx] = list(np.append(single_forward_table[min_idx], dcu_to_move))
                flow_each_dcu[max_idx] -= single_flow_table[dcu_to_move]
                flow_each_dcu[min_idx] += single_flow_table[dcu_to_move]

                # 如果连接数超过要求，从超过要求的dcu中拿出一个流量小的dcu
                if len(single_forward_table[min_idx]) >= max_link - self.n_gpu_per_group + 2:
                    flows = list()
                    for dcu_in in range(len(single_forward_table[min_idx])):
                        flows.append(single_flow_table[dcu_in])
                    sort_idx = np.argsort(flows)
                    rand_idx = np.random.randint(0, 10)
                    temp = single_forward_table[min_idx][sort_idx[rand_idx]]
                    single_forward_table[min_idx] = list(
                        np.delete(single_forward_table[min_idx], sort_idx[rand_idx]))
                    single_forward_table[max_idx] = list(np.append(single_forward_table[max_idx], temp))
                    flow_each_dcu[min_idx] -= single_flow_table[temp]
                    flow_each_dcu[max_idx] += single_flow_table[temp]

                # plt.clf()
                # plt.title('iter times=%d, max=%.6f, average=%.6f' %
                #           (j, np.max(flow_each_dcu), np.average(flow_each_dcu)))
                # plt.ylim(0.0003, 0.0008)
                # plt.plot(origin, color='blue', linestyle='--', alpha=0.4)
                # plt.plot(flow_each_dcu, color='blue')
                # plt.pause(0.001)

            # max_len = 0
            # for j in range(len(single_forward_table)):
            #     max_len = max(max_len, len(single_forward_table[j]))
            # print('max len = %d' % max_len)

            forward_table.append(single_forward_table)
            flow_table.append(flow_each_dcu)

        flow_array = list()
        for i in range(self.number_of_groups):
            for j in range(self.n_gpu_per_group):
                flow_array.append(flow_table[i][j])

        plt.plot(flow_array)
        plt.title('max=%d, average=%d' % (np.max(flow_array), np.average(flow_array)))
        plt.show()

        np.save(self.route_path + 'forwarding_table.npy', flow_array)

        return forward_table

    def cal_traffic_between_group_and_dcu(self, n_gpu_per_group):
        traffic_table_base_dcu = np.load(
            self.traffic_table_root + 'traffic_table_base_dcu_' + self.map_version + '.npy')

        traffic_group_to_dcu = np.zeros((n_gpu_per_group, self.N))

        start_time = time.time()
        for i in range(self.N):
            for j in range(self.N):
                group_idx = i % n_gpu_per_group
                traffic_group_to_dcu[group_idx][j] += traffic_table_base_dcu[j][i]
            self.show_progress(i, self.N, start_time)

        # plt.figure(figsize=(10, 6), dpi=200)
        # plt.plot(traffic_group_to_dcu)
        print()
        print(np.max(traffic_group_to_dcu), np.min(traffic_group_to_dcu), np.average(traffic_group_to_dcu))

        # np.save(self.route_path + 'traffic_group_to_dcu.npy', traffic_group_to_dcu)
        # print(self.route_path + 'traffic_group_to_dcu.npy saved.')

        return traffic_group_to_dcu

    def generate_forwarding_table_17280(self, number_of_group, dcu_per_group, max_link, max_rate):
        assert number_of_group * dcu_per_group == self.N

        # if not os.path.exists(self.route_path + 'traffic_group_to_dcu.npy'):
        #     self.cal_traffic_between_group_and_dcu(self.n_gpu_per_group)
        #
        # traffic_group_to_dcu = np.load(self.route_path + 'traffic_group_to_dcu.npy')

        traffic_group_to_dcu = self.cal_traffic_between_group_and_dcu(self.n_gpu_per_group)

        traffic_per_dcu = np.empty(0)
        forwarding_table = list()

        for i in range(dcu_per_group):
            single_traffic_table = traffic_group_to_dcu[i]

            single_forward_table = list()  # 生成初始解
            idx = 0
            for j in range(number_of_group):
                single_forward_table.append([])
                for k in range(dcu_per_group):
                    if idx != dcu_per_group * j + i:
                        single_forward_table[j].append(idx)
                    idx += 1

            # 计算每个dcu转发的流量
            level2_traffic_per_dcu = np.zeros(number_of_group)
            for j in range(number_of_group):
                for idx in single_forward_table[j]:
                    level2_traffic_per_dcu[j] += single_traffic_table[idx]

            # 迭代使转发的流量更加平均
            while np.max(level2_traffic_per_dcu) > np.average(traffic_group_to_dcu) * dcu_per_group * max_rate:
                max_idx = int(np.argmax(level2_traffic_per_dcu))
                min_idx = int(np.argmin(level2_traffic_per_dcu))

                # 从负责转发的流量最大的dcu中，随机选择一个其负责转发的dcu，放入流量最小的dcu的负责转发列表中去
                random_idx = np.random.randint(0, len(single_forward_table[max_idx]))
                dcu_to_move = single_forward_table[max_idx][random_idx]
                single_forward_table[max_idx] = list(np.delete(single_forward_table[max_idx], random_idx))
                single_forward_table[min_idx] = list(np.append(single_forward_table[min_idx], dcu_to_move))
                level2_traffic_per_dcu[max_idx] -= single_traffic_table[dcu_to_move]
                level2_traffic_per_dcu[min_idx] += single_traffic_table[dcu_to_move]

                # 如果连接数超过要求，从超过要求的dcu中拿出一个流量小的dcu
                if len(single_forward_table[min_idx]) >= max_link - number_of_group + 2:
                    flows = list()
                    for dcu_in in range(len(single_forward_table[min_idx])):
                        flows.append(single_traffic_table[dcu_in])
                    sort_idx = np.argsort(flows)
                    rand_idx = np.random.randint(0, 10)
                    temp = single_forward_table[min_idx][sort_idx[rand_idx]]
                    single_forward_table[min_idx] = list(
                        np.delete(single_forward_table[min_idx], sort_idx[rand_idx]))
                    single_forward_table[max_idx] = list(np.append(single_forward_table[max_idx], temp))
                    level2_traffic_per_dcu[min_idx] -= single_traffic_table[temp]
                    level2_traffic_per_dcu[max_idx] += single_traffic_table[temp]

            traffic_per_dcu = np.hstack((traffic_per_dcu, level2_traffic_per_dcu))
            forwarding_table.append(single_forward_table)

            link_numbers = list()
            for temp in single_forward_table:
                link_numbers.append(len(temp))
            # print('i: %d, max link number = %d' % (i, max(link_numbers)))
            # print(np.max(level2_traffic_per_dcu), np.average(traffic_group_to_dcu) * dcu_per_group)

        print('###############################')
        print('max / average = %.4f' % (np.max(traffic_per_dcu) / np.average(traffic_per_dcu)))
        # plt.figure(figsize=(12, 6), dpi=200)
        # plt.title('max = %.f, average = %.f, max/average = %.2f' %
        #           (np.max(traffic_per_dcu), np.average(traffic_per_dcu),
        #            np.max(traffic_per_dcu) / np.average(traffic_per_dcu)))
        # plt.ylim(30000000, 160000000)
        # plt.plot(traffic_per_dcu, linewidth=0.2)
        # plt.savefig(self.route_path + 'level2_traffic.png')
        # plt.show()

        # import pickle
        # with open(self.route_path + 'forwarding_table.pickle', 'wb') as f:
        #     pickle.dump(forwarding_table, f)
        # print(self.route_path + 'forwarding_table.pickle saved.')

        return forwarding_table

    # 计算用流量/带宽估算的卡之间的传输时间
    def cal_time_simulation_table(self):
        traffic_table_base_voxel = np.load(self.traffic_table_root + "traffic_table_base_voxel.npy")
        traffic_table_base_dcu = np.zeros((self.N, self.N))

        start_time = time.time()
        for dst in range(self.N):
            for src in range(self.N):
                # 计算两个dcu之间的流量
                traffic_between_dcu = 0
                for voxel_src in self.map_table[src]:
                    conn_number_esti_sum = 0
                    for voxel_dst in self.map_table[dst]:
                        conn_number_esti_sum += traffic_table_base_voxel[voxel_src][voxel_dst]

                    # [dsts, src]的连接数估计
                    conn_number = np.unique(np.random.choice(int(self.neuron_number * self.size[voxel_src]),
                                                             int(conn_number_esti_sum), replace=True)).shape[0]
                    # [dsts, srcs]的连接数估计
                    traffic_between_dcu += conn_number

                traffic_table_base_dcu[dst][src] = traffic_between_dcu
            # self.show_progress(dst, 2000, start_time)
        end_time = time.time()
        print("Calculation of time simulation table complete. %.2fs consumed." % (end_time - start_time))

        np.save(self.traffic_table_root + 'time_simulation.npy', traffic_table_base_dcu)

    # 计算映射后每张dcu中体素size之和
    def show_size_per_dcu(self):
        size_origin = list()
        size_now = list()
        for i in range(self.N):
            size_origin.append(np.sum(self.size[np.ix_(self.sequential_map_without_invalid_index[i])]))
            size_now.append(np.sum(self.size[np.ix_(self.map_table_without_invalid_idx[i])]))

        np.save('size_origin.npy', size_origin)
        np.save('size_now.npy', size_now)

        plt.figure(figsize=(10, 6), dpi=100)
        # plt.title('max=%.6f, min=%.6f, average=%.6f' % (np.max(size_now), np.min(size_now), np.average(size_now)))
        plt.xlabel('the number of GPUs')
        plt.ylabel('voxel size')
        # plt.ylim(0, 0.0005)
        plt.plot(size_origin, color='blue', linestyle='--', alpha=0.2, label='sum of voxel size\nbefore partitioning')
        plt.plot(size_now, color='blue', alpha=0.9, label='sum of voxel size\nafter partitioning')
        plt.legend(fontsize=13)
        # plt.savefig('size_before_and_after_map.png')
        plt.show()

        print('hey')

    def show_size_degree_new(self):
        origin_size_degree = np.zeros(self.N)
        size_degree = np.zeros(self.N)

        for gpu_idx in range(self.N):
            for cortical_idx in self.sequential_map_without_invalid_index[gpu_idx]:
                origin_size_degree[gpu_idx] += self.size_multi_degree[cortical_idx]

        for gpu_idx in range(self.N):
            for cortical_idx in self.map_table_without_invalid_idx[gpu_idx]:
                size_degree[gpu_idx] += self.size_multi_degree[cortical_idx]

        plt.figure(figsize=(10, 6), dpi=200)
        plt.title(
            self.map_version + ", size*degree max / average = %.6f" % (np.max(size_degree) / np.average(size_degree)))
        plt.plot(origin_size_degree, color='blue', label="before")
        plt.plot(size_degree, color='red', label="after")
        plt.legend(fontsize=15)
        plt.show()

    # 计算以dcu为单位的连接概率矩阵
    def cal_connection_table_base_dcu(self):
        connection_table_base_dcu = np.zeros((self.N, self.N))

        start_time = time.time()
        for dst in range(self.N):
            for src in range(self.N):
                connection_table_base_dcu[dst][src] = np.sum(self.conn[np.ix_(self.map_table[dst],
                                                                              self.map_table[src])])
            self.show_progress(dst, self.N, start_time)

        print("Nonzero Rate: %.2f" % (np.count_nonzero(connection_table_base_dcu) / (self.N ** 2 / 100)) + "%")
        np.save('connection_table_base_dcu_17280.npy', connection_table_base_dcu)

    # 计算dcu之间的连接矩阵
    def cal_binary_connection_table_base_dcu(self):
        file_name = '../tables/map_table/binary_connection_table_base_dcu_' + self.map_version + '.npy'
        binary_connection_table_base_dcu = np.zeros((self.N, self.N), dtype=bool)
        start_time = time.time()

        print('Begin Calculation...')
        for src in range(self.N):
            for dst in range(self.N):
                for cortical_out in self.map_table[src]:
                    temp = False
                    for cortical_in in self.map_table[dst]:
                        if self.conn[cortical_out][cortical_in] != 0:
                            binary_connection_table_base_dcu[src][dst] = 1
                            temp = True
                            break
                    if temp:
                        break

            self.show_progress(src, self.N, start_time)

        print("Nonzero Rate: %.2f" % (np.count_nonzero(binary_connection_table_base_dcu) / (self.N ** 2 / 100)) + "%")
        np.save(file_name, binary_connection_table_base_dcu)

    def show_out_in_traffic_per_dcu(self):
        self.N = 20000
        traffic_table_base_dcu = np.load(self.traffic_table_root + "traffic_table_base_dcu_" +
                                         self.map_version + ".npy")

        out_traffic_per_dcu = np.zeros(self.N)
        in_traffic_per_dcu = np.zeros(self.N)

        for i in range(self.N):
            out_traffic_per_dcu[i] = np.sum(traffic_table_base_dcu[:, i]) - traffic_table_base_dcu[i][i]
            in_traffic_per_dcu[i] = np.sum(traffic_table_base_dcu[i, :]) - traffic_table_base_dcu[i][i]
            # out_traffic_per_dcu[i] = np.sum(traffic_table_base_dcu[:, i])
            # in_traffic_per_dcu[i] = np.sum(traffic_table_base_dcu[i, :])

        X = np.arange(0, self.N)
        Y = np.full(self.N, np.average(out_traffic_per_dcu))

        plt.figure(figsize=(9, 6), dpi=200)
        plt.title('out: max = %d, average = %d, average = %.4f'
                  % (np.max(out_traffic_per_dcu), np.average(out_traffic_per_dcu),
                     np.max(out_traffic_per_dcu) / np.average(out_traffic_per_dcu)))
        plt.plot(X, out_traffic_per_dcu)
        plt.plot(X, Y, linewidth=3, label="average")
        plt.plot(X, 3 * Y, color="green", linewidth=3, label="3x average")
        # plt.plot(X, 4 * Y, color="red", linewidth=3, label="4x average")
        plt.legend(fontsize=15)
        plt.show()

        plt.figure(figsize=(9, 6), dpi=200)
        plt.title('in: max = %d, average = %d, average = %.4f'
                  % (np.max(in_traffic_per_dcu), np.average(in_traffic_per_dcu),
                     np.max(in_traffic_per_dcu) / np.average(in_traffic_per_dcu)))
        plt.plot(in_traffic_per_dcu)
        plt.plot(X, Y, linewidth=3, label="average")
        plt.legend(fontsize=15)
        plt.show()

        # np.save('map_sequential_out_traffic.npy', out_traffic_per_dcu)
        # np.save('map_sequential_in_traffic.npy', in_traffic_per_dcu)

    def compute_size_degree_split(self, map_table):
        pass


# NOTE(@lyh): added on 2022.2.16
class MapAnalysisParallel(GenerateMap, Parallelism):
    def __init__(self):
        super().__init__()
        # 计算的维度

        self.map_table_split_temp = []
        self.gpu_changed_idx = []
        self.step_sort_index = None
        self.N = 2000
        self.route_dict_table = None
        self.route_dict_table_file = None
        self.dimensions = 2
        # self.route_table_3_dim = np.load('route_demo_2_dim.npy')
        # 读入map表

        # self.map_table_split_file = '20000_v3/4720000_map_table_split.npy'

        # self.map_table_split = self.read_map_pkl(self.map_table_split_file)
        #

        # self.map_table_split_file = '/public/home/ssct005t/project/wml_istbi/tables/map_table/map_10000/map_table_split.npy'

        # self.map_table_split_file = 'map_10000_1.1_cortical_v2_without_invalid_idx.pkl'
        # self.map_table_split = np.load(self.map_table_split_file,
        #                                allow_pickle=True)
        self.map_table_split_file = "/public/home/ssct005t/project/wml_istbi/tables/map_table/map_2000/83experimentByInput/020000_map_table_split.npy"
        self.map_table_split = np.load(self.map_table_split_file,
                                       allow_pickle=True).item()

        # map的key可能是str
        if isinstance(self.map_table_split[0], list):
            self.map_table_split = self.read_map_dict_pkl(self.map_table_split_file)
        else:
            # pass
            map_table_split_int = {}
            for k, v in self.map_table_split.items():
                map_table_split_int[int(k)] = v
            self.map_table_split = map_table_split_int

        # self.map_table_split = np.load(self.map_table_split_file,
        #                                allow_pickle=True).item()

        # if isinstance(self.map_table_split[0], list):
        #     self.map_table_split = self.read_map_dict_pkl(self.map_table_split_file)
        # else:
        #     # pass
        #     map_table_split_int = {}
        #     for k, v in self.map_table_split.items():
        #         map_table_split_int[int(k)] = v
        #     self.map_table_split = map_table_split_int

        # 读入指定维度的路由表
        # self.route_dict_table_file = 'route_demo_3_dim.npy'
        # self.route_dict_table = np.load(self.route_dict_table_file)

        # self.route_dict_set_table = self.get_traffic_route_dict_set(self.route_dict_table, in_idx)

        self.route_dict_table_file = 'route_default_2dim_100_100.npy'
        self.route_dict_table = np.load(self.route_dict_table_file)
        # self.map_table = np.load("map_10000_1.05_cortical_v2_without_invalid_idx.pkl", allow_pickle=True)
        self.output_sum = None
        self.traffic_table = None

        self.step = 5
        self.gpu_idx = np.zeros((self.step,), dtype=int)
        self.map_table_split_old = None

        dcu_name = "cuda:" + str(self.rank % 4)
        # dcu_name = "cpu"
        self.device = torch.device(dcu_name)

        # from tensorflow import keras
        # from keras.models import load_model
        # self.model = load_model("../pop_cosr_expr/pop_cost_input_2.h5")
        if self.rank == self.master_rank:
            self.show_basic_information()
            self.show_initialize_info()

        self.comm.barrier()

    def compute_p2p_traffic_between_two_gpu(self, gpu_out_idx, gpu_in_idx):
        dcu_name = "cuda:" + str(self.rank % 4)
        device = torch.device(dcu_name if torch.cuda.is_available() else "cpu")

        traffic_gpu_to_gpu = 0
        for population_out_idx in self.map_table_split[gpu_out_idx].keys():
            traffic_src_to_dst = list()
            for population_in_idx in self.map_table_split[gpu_in_idx].keys():
                conn_number_estimate = self.neuron_number * self.size[population_in_idx] * \
                                       self.map_table_split[gpu_in_idx][population_in_idx] * self.degree[
                                           population_in_idx] * self.conn[population_in_idx][population_out_idx]
                traffic_src_to_dst.append(conn_number_estimate)

            sample_range = int(self.neuron_number * self.size[population_out_idx] * self.map_table_split[gpu_out_idx][
                population_out_idx])
            sample_times = int(np.sum(traffic_src_to_dst))
            # print()
            # print("Sample range: %d, Sample times: %d" % (sample_range, sample_times))

            traffic_src_to_gpu = torch.unique(
                torch.randint(0, sample_range, (sample_times,), device=device).clone()).numel()
            torch.cuda.empty_cache()
            traffic_gpu_to_gpu += traffic_src_to_gpu

        return traffic_gpu_to_gpu

    def sample(self, sample_range, sample_times, n_slice=50):
        data = list()

        for i in range(n_slice):

            random_sample = torch.randint(0, int(sample_range), (int(sample_times / n_slice),), device=self.device)
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            # temp = np.unique(random_sample.clone())

            # temp = np.unique(random_sample.cpu().numpy())
            temp = torch.unique(random_sample.clone())
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            data.append(temp)
            del temp
        # print("%.fMB VRAM allocated." % (torch.cuda.memory_allocated(device=device) / 1000000))
        new_data = torch.cat(data)
        new_data = torch.unique(new_data)
        traffic = torch.unique(new_data).numel()
        return traffic

    # 绘制结果图像
    def draw(self, input, figure_name):
        N = self.N
        x = np.arange(N)
        plt.figure(figsize=(10, 6), dpi=100)
        plt.title('max = %.2e, average = %.2e, max / average = %.2f' % (
            np.max(input), np.average(input), np.max(input) / np.average(input)), fontsize=15)
        plt.xlabel('Process ID', fontsize=20)
        plt.ylabel('Spike Count', fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        # plt.scatter(x, msg_cnt,s=13)
        plt.plot(x, input)
        plt.plot(x, np.full(N, np.average(input)), linestyle='--', color='black', linewidth=2, label='1x avg.')
        plt.plot(x, np.full(N, np.average(input) * 5), linestyle='--', color='red', linewidth=2, label='5x avg.')
        plt.legend(fontsize=20)
        #     plt.plot(x, np.full(N, np.max(input)),linestyle='-',color='black',linewidth=2)
        plt.savefig(figure_name)
        plt.close()
        # plt.show()

        print(np.max(input) / np.average(input))
        print(np.average(input))
        print(np.max(input))

    def check_traffic(self, dim, **traffic_table_base_gpu_arr):
        if dim == 1:
            traffic_table_base_gpu = traffic_table_base_gpu_arr[0]
            sum_row, sum_col = np.zeros(self.N), np.zeros(self.N)
            for i in range(self.N):
                inside_node_idx = np.arange(i // 4 * 4, i // 4 * 4 + 4)
                sum_row[i] = np.sum(traffic_table_base_gpu[i, :])
                sum_col[i] = np.sum(traffic_table_base_gpu[:, i]) - traffic_table_base_gpu[i][i]

                for idx in inside_node_idx:
                    sum_row[i] -= traffic_table_base_gpu[i][idx]
                    sum_col[i] -= traffic_table_base_gpu[idx][i]

            print('in: ', np.max(sum_row) / np.average(sum_row))
            print('out: ', np.max(sum_col) / np.average(sum_col))
        elif dim == 2:

            traffic_table_base_gpu_out = traffic_table_base_gpu_arr[0]
            # traffic_table_base_gpu_in = traffic_table_base_gpu_arr[0]
            sum_out, sum_in = np.zeros(self.N), np.zeros(self.N)
            for i in range(self.N):
                inside_node_idx = np.arange(i // 4 * 4, i // 4 * 4 + 4)
                sum_out[i] = np.sum(traffic_table_base_gpu_out[:, i]) - traffic_table_base_gpu_out[i][i]
                # sum_in[i] = np.sum(traffic_table_base_gpu_in[:, i])

                for idx in inside_node_idx:
                    sum_out[i] -= traffic_table_base_gpu_out[idx][i]
                    # sum_in[i] -= traffic_table_base_gpu_in[idx][i]
            #
            # print('in: ', np.max(sum_in) / np.average(sum_out))
            print('out: ', np.max(sum_out) / np.average(sum_in))
        else:
            pass

    def get_idx_array(self, idx):
        dou = int(idx / self.n_gpu_per_group)
        return np.arange(dou * self.n_gpu_per_group, (dou + 1) * self.n_gpu_per_group)

    def model_predict_traffic(self, sample_range, sample_times):
        sample_range = (sample_range - 362339.49627266) / 2956180.31065841
        sample_times = (sample_times - 646511.68795556) / 918756.79992516
        a = np.zeros((1, 2))
        a[0][0] = sample_range
        a[0][1] = sample_times
        return self.model.predict(a)

    def compute_2_dim_traffic_between_two_gpu(self, gpu_out_idx, gpu_in_idx):
        dcu_name = "cuda:" + str(self.rank % 4)
        device = torch.device(dcu_name if torch.cuda.is_available() else "cpu")

        traffic_gpu_to_gpu = 0
        # print('before swap')
        # print(self.map_table_split[5861])
        #
        # print(self.map_table_split[4949])
        #
        # print('after swap')
        # print(self.map_table_split_old[5861])
        #
        # print(self.map_table_split_old[4949])

        for population_out_idx in self.map_table_split[gpu_out_idx].keys():
            traffic_src_to_dst = list()
            for population_in_idx in self.map_table_split[gpu_in_idx].keys():
                tmp = int(population_in_idx * self.n + population_out_idx)
                if tmp in self.conn_dict:
                    conn_number_estimate = self.neuron_number * self.size[population_in_idx] * \
                                           self.map_table_split[gpu_in_idx][population_in_idx] * self.degree[
                                               population_in_idx] * self.conn_dict[tmp]
                    traffic_src_to_dst.append(conn_number_estimate)
                else:
                    traffic_src_to_dst.append(0.0)

            sample_range = int(self.neuron_number * self.size[population_out_idx] * self.map_table_split[gpu_out_idx][
                population_out_idx])
            sample_times = int(np.sum(traffic_src_to_dst))
            # print()
            # print("Sample range: %d, Sample times: %d" % (sample_range, sample_times))

            torch.cuda.empty_cache()
            n = 1
            if sample_range > 1e7 or sample_times > 1e8:
                n = 30
            traffic_src_to_gpu = self.sample(sample_range, sample_times, n_slice=n)
            torch.cuda.empty_cache()
            # if self.rank == 1:
            #     print("计算gpu to gpu")
            # traffic_src_to_gpu = self.model_predict_traffic(sample_range, sample_times)
            # if self.rank == 1:
            #     print(traffic_src_to_gpu)
            traffic_gpu_to_gpu += traffic_src_to_gpu

        # output.append(traffic_src_to_dcu)
        return traffic_gpu_to_gpu

    def compute_2_dim_traffic_between_gpu_and_group(self, gpu_out_idx, gpu_in_idx):
        dcu_name = "cuda:" + str(self.rank % 4)
        device = torch.device(dcu_name if torch.cuda.is_available() else "cpu")

        traffic_gpu_to_gpu = 0

        for population_out_idx in self.map_table_split[gpu_out_idx].keys():
            traffic_src_to_dst = list()
            # gpu_in_idx_arr = self.get_idx_array(gpu_in_idx)
            # st = gpu_in_idx * self.n_gpu_per_group
            # en = st + self.n_gpu_per_group
            for in_idx in gpu_in_idx:
                for population_in_idx in self.map_table_split[int(in_idx)].keys():
                    tmp = int(population_in_idx * self.n + population_out_idx)
                    if tmp in self.conn_dict:
                        conn_number_estimate = self.neuron_number * self.size[population_in_idx] * \
                                               self.map_table_split[in_idx][population_in_idx] * self.degree[
                                                   population_in_idx] * self.conn_dict[tmp]
                        traffic_src_to_dst.append(conn_number_estimate)
                    else:
                        traffic_src_to_dst.append(0.0)

            sample_range = int(self.neuron_number * self.size[population_out_idx] * self.map_table_split[gpu_out_idx][
                population_out_idx])
            sample_times = int(np.sum(traffic_src_to_dst))
            # print()
            # print("Sample range: %d, Sample times: %d" % (sample_range, sample_times))

            torch.cuda.empty_cache()
            n = 1
            if sample_range > 1e7 or sample_times > 1e8:
                n = 30
            traffic_src_to_gpu = self.sample(sample_range, sample_times, n_slice=n)
            torch.cuda.empty_cache()

            # traffic_src_to_gpu = self.model_predict_traffic(sample_range, sample_times)
            traffic_gpu_to_gpu += traffic_src_to_gpu

        # output.append(traffic_src_to_dcu)
        return traffic_gpu_to_gpu

    # 根据route_table得到易于计算out_traffic的route_table
    def get_traffic_route_dict_set(self, row):
        route_dict = {}
        for i in range(20000):
            if self.route_dict_table[row][i] in route_dict:
                route_dict[self.route_dict_table[row][i]].add(i)
            else:
                route_dict[self.route_dict_table[row][i]] = {i}
        for key in route_dict.keys():
            if key == row:
                route_dict[key].remove(key)
            else:
                route_dict[row].remove(key)
                route_dict[key].add(key)
        return route_dict

    def calculate_one_dcu_output_traffic(self, out_idx):
        output_first_stage = 0.0
        output_second_stage = 0.0

        route_dict_set_table = self.get_traffic_route_dict_set(out_idx)
        # 第一阶段，第一部分：计算可以直接到达的流量
        for in_idx in route_dict_set_table[out_idx]:
            output_first_stage += self.compute_2_dim_traffic_between_two_gpu(out_idx, in_idx)

        # for in_idx in row_idx:
        #     output_first_stage += self.compute_2_dim_traffic_between_two_gpu(out_idx, in_idx)

        # 第一阶段，第二部分：计算并包发送的流量
        for key in route_dict_set_table.keys():
            if key == out_idx:
                continue
            output_first_stage += self.compute_2_dim_traffic_between_gpu_and_group(out_idx, route_dict_set_table[key])

            # 第二阶段
            for in_idx in route_dict_set_table[out_idx]:
                output_second_stage += self.compute_2_dim_traffic_between_two_gpu(key, in_idx)

        return output_first_stage, output_second_stage

    def calculate_one_dcu_input_traffic(self, in_idx):
        input_first_stage = 0.0
        input_second_stage = 0.0
        route_dict_set_table = self.get_traffic_route_dict_set(in_idx)
        # send_row_route_set = route_dict_set_table[in_idx]
        send_route_set = route_dict_set_table[in_idx].add(in_idx)

        # 第一阶段,第一部分：计算并包的流量
        for key in route_dict_set_table:
            if key == in_idx:
                continue
            input_first_stage += self.compute_2_dim_traffic_between_gpu_and_group(key, route_dict_set_table[in_idx])
            # 第二阶段
            for out_idx in route_dict_set_table[key]:
                if out_idx == key:
                    continue
                input_second_stage += self.compute_2_dim_traffic_between_two_gpu(out_idx, in_idx)
        # 第一阶段,第二部分：计算同一组的其余流量
        for out_idx in route_dict_set_table[in_idx]:
            input_first_stage += self.compute_2_dim_traffic_between_two_gpu(out_idx, in_idx)
        # 第二阶段

        # 第一阶段接受的流量

        # input = np.zeros((self.N,))
        # input_traffic = 0
        # for out_idx in range(self.N):
        #     if out_idx == in_idx:
        #         continue
        #     if out_idx % self.n_gpu_per_group == in_idx % self.n_gpu_per_group:
        #         #  处于同列
        #         input_first_stage += self.compute_2_dim_traffic_between_gpu_and_group(out_idx, in_idx)
        #     elif int(out_idx / self.n_gpu_per_group) == int(in_idx / self.n_gpu_per_group):
        #         input_first_stage += self.compute_2_dim_traffic_between_two_gpu(out_idx, in_idx)
        #     else:
        #         input_second_stage += self.compute_2_dim_traffic_between_two_gpu(out_idx, in_idx)
        return input_first_stage, input_second_stage

    # 当以p2p方式进行通信时，每个进程发送/接收的流量之和
    def compute_1_dim_traffic(self):
        self.route_dict_table_file = 'route_default_1dim_800.npy'
        self.route_dict_table = np.load(self.route_dict_table_file)
        if self.rank == self.master_rank:
            time_start = time.time()
            traffic_table_base_gpu_out_in = np.zeros((self.N, 2))

            for col_idx in range(self.N):
                time1 = time.time()
                msg_out_in = self.comm.recv(source=(col_idx % (self.comm_size - 1)))
                traffic_table_base_gpu_out_in += msg_out_in
                # traffic_table_base_gpu[col_idx] = msg
                time2 = time.time()
                # print('Col %d: %.4fs consumed.' % (col_idx, time2 - time1))

            time.sleep(10)

            file_path = self.traffic_table_root + self.map_table_split_file[0:len(
                self.map_table_split_file) - 4] + '/' + self.route_dict_table_file[
                                                        0:len(self.route_dict_table_file) - 4] + '/'
            if not os.path.exists(file_path):
                os.makedirs(file_path)

            npy_name = "traffic_table_base_dcu_out_in_1_dim" + self.map_version + ".npy"
            np.save(file_path + npy_name, traffic_table_base_gpu_out_in)
            print(file_path + npy_name + " saved.")

            traffic_table_base_gpu_out_in_t = traffic_table_base_gpu_out_in.T
            figure_name = file_path + "traffic_table_out_sum.png"
            self.draw(traffic_table_base_gpu_out_in_t[0], figure_name)
            figure_name = file_path + "traffic_table_in_sum.png"
            self.draw(traffic_table_base_gpu_out_in_t[1], figure_name)

        else:
            column_idx_to_process = self.allocate_idx_to_calculate()

            for gpu_out_idx in column_idx_to_process:
                traffic_table_base_gpu_out_tmp = np.zeros((self.N, 1))
                traffic_table_base_gpu_in_tmp = np.zeros((self.N, 1))

                time1 = time.time()
                for gpu_in_idx in range(self.N):
                    tmp = self.compute_2_dim_traffic_between_two_gpu(gpu_out_idx, gpu_in_idx)
                    traffic_table_base_gpu_out_tmp[gpu_out_idx] += tmp
                    traffic_table_base_gpu_in_tmp[gpu_in_idx] += tmp

                time2 = time.time()
                traffic_table_base_gpu_out_in = np.concatenate(
                    (traffic_table_base_gpu_out_tmp, traffic_table_base_gpu_in_tmp), axis=1)
                self.comm.send(traffic_table_base_gpu_out_in, dest=self.master_rank)
                print('Col %d sent: %.4fs consumed.' % (gpu_out_idx, time2 - time1))

    def get_route_dict_out(self, out_idx, in_idx_list):
        route_dict_out_idx = {}
        for idx in in_idx_list:
            if idx == out_idx:
                continue
            else:
                # 直接发送
                if self.route_dict_table[out_idx][idx] == out_idx and not (idx in route_dict_out_idx):
                    route_dict_out_idx[int(idx)] = {idx}
                else:
                    if not (self.route_dict_table[out_idx][idx] in route_dict_out_idx):
                        route_dict_out_idx[int(self.route_dict_table[out_idx][idx])] = {
                            self.route_dict_table[out_idx][idx]}
                    route_dict_out_idx[int(self.route_dict_table[out_idx][idx])].add(idx)
        route_dict_out_idx_tmp = {}
        for k, v in route_dict_out_idx.items():
            if out_idx in v and len(v) > 1:
                continue
            route_dict_out_idx_tmp[int(k)] = v
        return route_dict_out_idx_tmp

    def calculate_2_dim_input_output_traffic(self, idx):
        self.dimensions = 2
        tem_output_traffic = np.zeros((self.N, self.dimensions))
        tem_input_traffic = np.zeros((self.N, self.dimensions))
        route_dict_out_idx = self.get_route_dict_out(idx, range(self.N))
        for in_idx, in_idx_list in route_dict_out_idx.items():
            stage = 0
            if len(in_idx_list) == 1:
                if not self.is_in_same_node([idx, in_idx]):
                    tmp = self.compute_2_dim_traffic_between_two_gpu(idx, in_idx)
                    tem_output_traffic[idx][stage] += tmp
                    tem_input_traffic[in_idx][stage] += tmp
            else:
                tmp = self.compute_2_dim_traffic_between_gpu_and_group(idx, in_idx_list)
                tem_output_traffic[idx][stage] += tmp
                tem_input_traffic[in_idx][stage] += tmp
                route_dict_out_idx_tmp_1 = self.get_route_dict_out(in_idx, in_idx_list)
                for in_idx_1, in_idx_list_1 in route_dict_out_idx_tmp_1.items():
                    if not self.is_in_same_node([idx, in_idx_1]):
                        stage = 1
                        tmp = self.compute_2_dim_traffic_between_two_gpu(idx, in_idx_1)
                        tem_output_traffic[int(in_idx)][stage] += tmp
                        tem_input_traffic[int(in_idx_1)][stage] += tmp
        return tem_output_traffic, tem_input_traffic

        # 当以2级虚拟拓扑进行通信时，每个进程1/2级的发送/接收流量之和

    def compute_2_dim_traffic(self):
        self.route_dict_table_file = '/public/home/ssct005t/project/wml_istbi/code/generate_route/route_default_2dim_40_50.npy'
        self.route_dict_table = np.load(self.route_dict_table_file)

        if self.rank == self.master_rank:
            time_start = time.time()
            traffic_table_base_gpu_out_in = np.zeros((self.N, 4))
            # traffic_table_base_gpu_in = np.zeros((self.N, self.N))

            for col_idx in range(self.N):
                time1 = time.time()
                msg_out_in = self.comm.recv(source=(col_idx % (self.comm_size - 1)))
                traffic_table_base_gpu_out_in += msg_out_in

                time2 = time.time()
                # print('Col %d: %.4fs consumed.' % (col_idx, time2 - time1))

            time.sleep(10)

            # 建立二维的路径
            file_path = self.traffic_table_root + self.map_table_split_file[0:len(
                self.map_table_split_file) - 4] + '/' + self.route_dict_table_file[
                                                        0:len(self.route_dict_table_file) - 4] + '/'
            if not os.path.exists(file_path):
                os.makedirs(file_path)

            npy_name = "traffic_table_base_dcu_out_in_2_dim" + self.map_version + ".npy"
            np.save(file_path + npy_name, traffic_table_base_gpu_out_in)
            print(file_path + npy_name + " saved.")
            traffic_table_base_gpu_out_in_t = traffic_table_base_gpu_out_in.T
            for i in range(4):
                if i < 2:
                    figure_name = file_path + 'traffic_table_out_' + str(i) + "stage" + '.png'
                else:
                    figure_name = file_path + 'traffic_table_in_' + str(i - 2) + "stage" + '.png'
                self.draw(traffic_table_base_gpu_out_in_t[i], figure_name)

            figure_name = file_path + "traffic_table_out_sum.png"
            self.draw(traffic_table_base_gpu_out_in_t[0] + traffic_table_base_gpu_out_in_t[1], figure_name)
            figure_name = file_path + "traffic_table_in_sum.png"
            self.draw(traffic_table_base_gpu_out_in_t[2] + traffic_table_base_gpu_out_in_t[3], figure_name)

            # npy_name = "traffic_table_base_dcu_in" + self.map_version + ".npy"
            # np.save(self.traffic_table_root + npy_name, traffic_table_base_gpu_in)
            # print(self.traffic_table_root + npy_name + " saved.")

            time_end = time.time()
            print("%d nodes used. " % ((self.comm_size - 1) / 8 + 1))
            print("%.f seconds consumed." % (time_end - time_start))


        else:
            column_idx_to_process = self.allocate_idx_to_calculate()
            for gpu_out_idx in column_idx_to_process:
                # calculate the traffic, stored as a (N, 1) array
                # traffic_base_gpu_for_a_column_out = np.zeros((self.N,))
                traffic_table_base_gpu_out_tmp = np.zeros((self.N, 2))
                traffic_table_base_gpu_in_tmp = np.zeros((self.N, 2))
                # traffic_base_gpu_for_a_column_in = np.zeros((self.N,))
                time1 = time.time()
                traffic_table_base_gpu_out_tmp, traffic_table_base_gpu_in_tmp = self.calculate_2_dim_input_output_traffic(
                    gpu_out_idx)
                time2 = time.time()
                traffic_table_base_gpu_out_in = np.concatenate(
                    (traffic_table_base_gpu_out_tmp, traffic_table_base_gpu_in_tmp), axis=1)
                self.comm.send(traffic_table_base_gpu_out_in, dest=self.master_rank)
                # self.comm.send(traffic_base_gpu_for_a_column_in, dest=self.master_rank)
                print('Col %d sent: %.4fs consumed.' % (gpu_out_idx, time2 - time1))

    def get_3_dim_route_dict_out(self, out_idx, in_idx_list):
        route_dict_out_idx = {}
        for idx in in_idx_list:
            if idx == out_idx:
                continue
            else:
                # 直接发送
                if self.route_dict_table[out_idx][idx] == out_idx and not (idx in route_dict_out_idx):
                    route_dict_out_idx[int(idx)] = {idx}
                else:
                    if not (self.route_dict_table[out_idx][idx] in route_dict_out_idx):
                        route_dict_out_idx[int(self.route_dict_table[out_idx][idx])] = {
                            self.route_dict_table[out_idx][idx]}
                    route_dict_out_idx[int(self.route_dict_table[out_idx][idx])].add(idx)

        return route_dict_out_idx

    # def get_4_dim_route_dict_out(self, out_idx, in_idx_list):
    #     route_dict_out_idx = {}
    #     for idx in in_idx_list:
    #         if idx == out_idx:
    #             continue
    #         else:
    #             if self.route_dict_table[out_idx][idx] in route_dict_out_idx:
    #                 route_dict_out_idx[int(self.route_dict_table[out_idx][idx])].add(idx)
    #             else:
    #                 route_dict_out_idx[int(self.route_dict_table[out_idx][idx])] = {idx}
    #     return route_dict_out_idx

    # 没有用途的函数
    def calculate_3_dim_one_dcu_output_traffic_one_stage(self, out_idx, in_idx_list, stage):
        tem_output_traffic = np.zeros((self.N, self.dimensions))
        route_dict_out_idx = self.get_3_dim_route_dict_out(out_idx, in_idx_list)
        for in_idx, in_idx_list in route_dict_out_idx.items():
            if len(in_idx_list) == 1:
                tem_output_traffic[out_idx][stage] += self.compute_2_dim_traffic_between_two_gpu(out_idx, in_idx)
            else:
                tem_output_traffic[out_idx][stage] += self.compute_2_dim_traffic_between_gpu_and_group(out_idx,
                                                                                                       in_idx_list)
        return tem_output_traffic

    def calculate_3_dim_input_output_traffic(self, idx):
        self.dimensions = 3
        tem_output_traffic = np.zeros((self.N, self.dimensions))
        tem_input_traffic = np.zeros((self.N, self.dimensions))

        route_dict_out_idx = self.get_3_dim_route_dict_out(idx, range(self.N))

        for in_idx, in_idx_list in route_dict_out_idx.items():
            stage = 0
            # 直接传输
            if len(in_idx_list) == 1:
                tmp = self.compute_2_dim_traffic_between_two_gpu(idx, in_idx)
                tem_output_traffic[idx][stage] += tmp
                tem_input_traffic[in_idx][stage] += tmp
            else:
                tmp = self.compute_2_dim_traffic_between_gpu_and_group(idx, in_idx_list)
                tem_output_traffic[idx][stage] += tmp
                tem_input_traffic[in_idx][stage] += tmp
                route_dict_out_idx_tmp_1 = self.get_3_dim_route_dict_out(in_idx, in_idx_list)
                for in_idx_1, in_idx_list_1 in route_dict_out_idx_tmp_1.items():
                    stage = 1
                    if len(in_idx_list_1) == 1:
                        tmp = self.compute_2_dim_traffic_between_two_gpu(idx, in_idx_1)
                        tem_output_traffic[int(in_idx)][stage] += tmp
                        tem_input_traffic[int(in_idx_1)][stage] += tmp
                    else:
                        tmp = self.compute_2_dim_traffic_between_gpu_and_group(idx, in_idx_list_1)
                        tem_output_traffic[int(in_idx)][stage] += tmp
                        tem_input_traffic[int(in_idx)][stage] += tmp
                        route_dict_out_idx_tmp_2 = self.get_3_dim_route_dict_out(in_idx_1, in_idx_list_1)
                        for in_idx_2, in_idx_list_2 in route_dict_out_idx_tmp_2.items():
                            stage = 2
                            tmp = self.compute_2_dim_traffic_between_two_gpu(idx, in_idx_2)
                            tem_output_traffic[in_idx_1][stage] += tmp
                            tem_input_traffic[in_idx_2][stage] += tmp
        return tem_output_traffic, tem_input_traffic

    def calculate_3_dim_one_dcu_output_traffic(self, out_idx):
        # 一共N个进程，每个进程三个阶段
        tem_output_traffic = np.zeros((self.N, self.dimensions))

        route_dict_out_idx = self.get_3_dim_route_dict_out(out_idx, range(self.N))

        for in_idx, in_idx_list in route_dict_out_idx.items():
            stage = 0
            # 直接传输
            if len(in_idx_list) == 1:
                tem_output_traffic[out_idx][stage] += self.compute_2_dim_traffic_between_two_gpu(out_idx, in_idx)
            else:
                tem_output_traffic[out_idx][stage] += self.compute_2_dim_traffic_between_gpu_and_group(out_idx,
                                                                                                       in_idx_list)
                route_dict_out_idx_tmp_1 = self.get_3_dim_route_dict_out(in_idx, in_idx_list)
                for in_idx_1, in_idx_list_1 in route_dict_out_idx_tmp_1.items():
                    stage = 1
                    if len(in_idx_list_1) == 1:
                        tem_output_traffic[int(in_idx)][stage] += self.compute_2_dim_traffic_between_two_gpu(out_idx,
                                                                                                             in_idx_1)
                    else:
                        tem_output_traffic[int(in_idx)][stage] += self.compute_2_dim_traffic_between_gpu_and_group(
                            out_idx,
                            in_idx_list_1)
                        route_dict_out_idx_tmp_2 = self.get_3_dim_route_dict_out(in_idx_1, in_idx_list_1)
                        for in_idx_2, in_idx_list_2 in route_dict_out_idx_tmp_2.items():
                            stage = 2
                            tem_output_traffic[in_idx_1][stage] += self.compute_2_dim_traffic_between_two_gpu(out_idx,
                                                                                                              in_idx_2)
        return tem_output_traffic

    def calculate_3_dim_one_dcu_input_traffic(self, idx):
        # 一共N个进程，每个进程三个阶段
        tem_input_traffic = np.zeros((self.N, self.dimensions))

        route_dict_out_idx = self.get_3_dim_route_dict_out(idx, range(self.N))

        for in_idx, in_idx_list in route_dict_out_idx.items():
            stage = 0
            # 直接传输
            if len(in_idx_list) == 1:
                tem_input_traffic[in_idx][stage] += self.compute_2_dim_traffic_between_two_gpu(idx, in_idx)
            else:
                tem_input_traffic[in_idx][stage] += self.compute_2_dim_traffic_between_gpu_and_group(idx,
                                                                                                     in_idx_list)
                route_dict_out_idx_tmp_1 = self.get_3_dim_route_dict_out(in_idx, in_idx_list)
                for in_idx_1, in_idx_list_1 in route_dict_out_idx_tmp_1.items():
                    stage = 1
                    if len(in_idx_list_1) == 1:
                        tem_input_traffic[int(in_idx_1)][stage] += self.compute_2_dim_traffic_between_two_gpu(idx,
                                                                                                              in_idx_1)
                    else:
                        tem_input_traffic[int(in_idx)][stage] += self.compute_2_dim_traffic_between_gpu_and_group(
                            idx,
                            in_idx_list_1)
                        route_dict_out_idx_tmp_2 = self.get_3_dim_route_dict_out(in_idx_1, in_idx_list_1)
                        for in_idx_2, in_idx_list_2 in route_dict_out_idx_tmp_2.items():
                            stage = 2
                            tem_input_traffic[in_idx_2][stage] += self.compute_2_dim_traffic_between_two_gpu(idx,
                                                                                                             in_idx_2)
        return tem_input_traffic

    # 当以3级虚拟拓扑进行通信时，每个进程1/2/3的发送/接收流量之和
    def compute_3_dim_traffic(self):
        self.route_dict_table_file = 'route_default_3dim_8_10_10.npy'
        self.route_dict_table = np.load(self.route_dict_table_file)
        if self.rank == self.master_rank:
            time_start = time.time()
            # traffic_table_base_gpu_out = np.zeros((self.N, 3))
            # traffic_table_base_gpu_in = np.zeros((self.N, 3))
            traffic_table_base_gpu_out_in = np.zeros((self.N, 6))

            for col_idx in range(self.N):
                time1 = time.time()
                msg_out_in = self.comm.recv(source=(col_idx % (self.comm_size - 1)))
                traffic_table_base_gpu_out_in += msg_out_in
                # traffic_table_base_gpu_out += msg_out
                # msg_in = self.comm.recv(source=(col_idx % (self.comm_size - 1)))
                # traffic_table_base_gpu_in += msg_in
                time2 = time.time()
                # print('Col %d: %.4fs consumed.' % (col_idx, time2 - time1))

            time.sleep(10)
            # 建立三维的路径
            file_path = self.traffic_table_root + self.map_table_split_file[0:len(
                self.map_table_split_file) - 4] + '/' + self.route_dict_table_file[
                                                        0:len(self.route_dict_table_file) - 4] + '/'
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            npy_name = "traffic_table_base_dcu_out_in_3_dim" + self.map_version + ".npy"
            np.save(file_path + npy_name, traffic_table_base_gpu_out_in)
            print(file_path + npy_name + " saved.")
            traffic_table_base_gpu_out_in_t = traffic_table_base_gpu_out_in.T
            for i in range(6):
                if i < 3:
                    figure_name = file_path + 'traffic_table_out_' + str(i) + "stage" + '.png'
                else:
                    figure_name = file_path + 'traffic_table_in_' + str(i - 3) + "stage" + '.png'
                self.draw(traffic_table_base_gpu_out_in_t[i], figure_name)

            figure_name = file_path + "traffic_table_out_sum.png"
            self.draw(traffic_table_base_gpu_out_in_t[0] + traffic_table_base_gpu_out_in_t[1] +
                      traffic_table_base_gpu_out_in_t[2], figure_name)
            figure_name = file_path + "traffic_table_in_sum.png"
            self.draw(traffic_table_base_gpu_out_in_t[3] + traffic_table_base_gpu_out_in_t[4] +
                      traffic_table_base_gpu_out_in_t[5], figure_name)

            # npy_name = "traffic_table_base_dcu_out_3_dim" + self.map_version + ".npy"
            # np.save(self.traffic_table_root + npy_name, traffic_table_base_gpu_out)
            #
            # npy_name = "traffic_table_base_dcu_in_3_dim" + self.map_version + ".npy"
            # np.save(self.traffic_table_root + npy_name, traffic_table_base_gpu_in)

            # npy_name = "traffic_table_base_dcu_in" + self.map_version + ".npy"
            # np.save(self.traffic_table_root + npy_name, traffic_table_base_gpu_in)
            # print(self.traffic_table_root + npy_name + " saved.")

            time_end = time.time()
            print("%d nodes used. " % ((self.comm_size - 1) / 8 + 1))
            print("%.f seconds consumed." % (time_end - time_start))

            # self.check_traffic(2, traffic_table_base_gpu_out, traffic_table_base_gpu_in)
            # self.check_traffic(2, traffic_table_base_gpu_out)
        else:
            column_idx_to_process = self.allocate_idx_to_calculate()
            for gpu_out_idx in column_idx_to_process:
                traffic_table_base_gpu_out_tmp = np.zeros((self.N, 3))
                traffic_table_base_gpu_in_tmp = np.zeros((self.N, 3))
                # traffic_base_gpu_for_a_column_in = np.zeros((self.N,))
                time1 = time.time()
                # output_first_stage, output_second_stage = self.calculate_one_dcu_output_traffic(gpu_out_idx)

                # traffic_table_base_gpu_out_tmp = self.calculate_3_dim_one_dcu_output_traffic(gpu_out_idx)
                # traffic_table_base_gpu_in_tmp = self.calculate_3_dim_one_dcu_input_traffic(gpu_out_idx)
                traffic_table_base_gpu_out_tmp, traffic_table_base_gpu_in_tmp = self.calculate_3_dim_input_output_traffic(
                    gpu_out_idx)

                time2 = time.time()
                traffic_table_base_gpu_out_in = np.concatenate(
                    (traffic_table_base_gpu_out_tmp, traffic_table_base_gpu_in_tmp), axis=1)
                self.comm.send(traffic_table_base_gpu_out_in,
                               dest=self.master_rank)

                # self.comm.send(traffic_table_base_gpu_in_tmp, dest=self.master_rank)
                # self.comm.send(traffic_base_gpu_for_a_column_in, dest=self.master_rank)
                print('Col %d sent: %.4fs consumed.' % (gpu_out_idx, time2 - time1))

    # def get_4_dim_route_dict_out(self, out_idx, in_idx_list):
    #     route_dict_out_idx = {}
    #     for idx in in_idx_list:
    #         if idx == out_idx:
    #             continue
    #         else:
    #             if self.route_dict_table[out_idx][idx] in route_dict_out_idx:
    #                 route_dict_out_idx[int(self.route_dict_table[out_idx][idx])].add(idx)
    #             else:
    #                 route_dict_out_idx[int(self.route_dict_table[out_idx][idx])] = {idx}
    #     return route_dict_out_idx

    def get_4_dim_route_dict_out(self, out_idx, in_idx_list):
        route_dict_out_idx = {}
        for idx in in_idx_list:
            if idx == out_idx:
                continue
            else:
                # 直接发送
                if self.route_dict_table[out_idx][idx] == out_idx and not (idx in route_dict_out_idx):
                    route_dict_out_idx[int(idx)] = {idx}
                else:
                    if not (self.route_dict_table[out_idx][idx] in route_dict_out_idx):
                        route_dict_out_idx[int(self.route_dict_table[out_idx][idx])] = {
                            self.route_dict_table[out_idx][idx]}
                    route_dict_out_idx[int(self.route_dict_table[out_idx][idx])].add(idx)

        return route_dict_out_idx

    def calculate_4_dim_one_dcu_output_traffic(self, idx):
        # 一共N个进程，每个进程四个阶段
        self.dimensions = 4
        tem_output_traffic = np.zeros((self.N, self.dimensions))
        tem_input_traffic = np.zeros((self.N, self.dimensions))

        route_dict_out_idx = self.get_4_dim_route_dict_out(idx, range(self.N))

        for in_idx, in_idx_list in route_dict_out_idx.items():
            stage = 0
            # 直接传输
            if len(in_idx_list) == 1:
                tmp = self.compute_2_dim_traffic_between_two_gpu(idx, in_idx)
                tem_output_traffic[idx][stage] += tmp
                tem_input_traffic[in_idx][stage] += tmp
            else:
                tmp = self.compute_2_dim_traffic_between_gpu_and_group(idx, in_idx_list)
                tem_output_traffic[idx][stage] += tmp
                tem_input_traffic[in_idx][stage] += tmp
                route_dict_out_idx_tmp_1 = self.get_4_dim_route_dict_out(in_idx, in_idx_list)
                for in_idx_1, in_idx_list_1 in route_dict_out_idx_tmp_1.items():
                    stage = 1
                    if len(in_idx_list_1) == 1:
                        tmp = self.compute_2_dim_traffic_between_two_gpu(idx, in_idx_1)
                        tem_output_traffic[int(in_idx)][stage] += tmp
                        tem_input_traffic[int(in_idx_1)][stage] += tmp
                    else:
                        tmp = self.compute_2_dim_traffic_between_gpu_and_group(idx, in_idx_list_1)
                        tem_output_traffic[int(in_idx)][stage] += tmp
                        tem_input_traffic[int(in_idx)][stage] += tmp
                        route_dict_out_idx_tmp_2 = self.get_4_dim_route_dict_out(in_idx_1, in_idx_list_1)
                        for in_idx_2, in_idx_list_2 in route_dict_out_idx_tmp_2.items():
                            stage = 2
                            if len(in_idx_list_2) == 1:
                                tmp = self.compute_2_dim_traffic_between_two_gpu(idx, in_idx_2)
                                tem_output_traffic[int(in_idx_1)][stage] += tmp
                                tem_input_traffic[in_idx_2][stage] += tmp
                            else:
                                tmp = self.compute_2_dim_traffic_between_gpu_and_group(idx, in_idx_list_2)
                                tem_output_traffic[int(in_idx_1)][stage] += tmp
                                tem_input_traffic[int(in_idx_1)][stage] += tmp
                                route_dict_out_idx_tmp_3 = self.get_4_dim_route_dict_out(in_idx_2, in_idx_list_2)
                                for in_idx_3, in_idx_list_3 in route_dict_out_idx_tmp_3.items():
                                    stage = 3
                                    tmp = self.compute_2_dim_traffic_between_two_gpu(idx, in_idx_3)
                                    tem_output_traffic[int(in_idx_2)][stage] += tmp
                                    tem_input_traffic[int(in_idx_3)][stage] += tmp

        return tem_output_traffic, tem_input_traffic

    def compute_4_dim_traffic(self):
        self.route_dict_table_file = 'route_default_4dim_8_5_5_4.npy'
        self.route_dict_table = np.load(self.route_dict_table_file)
        if self.rank == self.master_rank:
            time_start = time.time()
            traffic_table_base_gpu_out_in = np.zeros((self.N, 8))
            # traffic_table_base_gpu_in = np.zeros((self.N, 4))

            for col_idx in range(self.N):
                time1 = time.time()
                msg_out_in = self.comm.recv(source=(col_idx % (self.comm_size - 1)))
                traffic_table_base_gpu_out_in += msg_out_in
                # msg_in = self.comm.recv(source=(col_idx % (self.comm_size - 1)))
                # traffic_table_base_gpu_in += msg_in

                time2 = time.time()
                # print('Col %d: %.4fs consumed.' % (col_idx, time2 - time1))

            time.sleep(10)
            file_path = self.traffic_table_root + self.map_table_split_file[0:len(
                self.map_table_split_file) - 4] + '/' + self.route_dict_table_file[
                                                        0:len(self.route_dict_table_file) - 4] + '/'
            if not os.path.exists(file_path):
                os.makedirs(file_path)

            npy_name = "traffic_table_base_dcu_out_in_4_dim" + self.map_version + ".npy"
            np.save(file_path + npy_name, traffic_table_base_gpu_out_in)
            print(file_path + npy_name + " saved.")
            # npy_name = "traffic_table_base_dcu_in_3_dim" + self.map_version + ".npy"
            # np.save(self.traffic_table_root + npy_name, traffic_table_base_gpu_in)
            # print(self.traffic_table_root + npy_name + " saved.")

            # npy_name = "traffic_table_base_dcu_in" + self.map_version + ".npy"
            # np.save(self.traffic_table_root + npy_name, traffic_table_base_gpu_in)
            # print(self.traffic_table_root + npy_name + " saved.")
            traffic_table_base_gpu_out_in_t = traffic_table_base_gpu_out_in.T
            for i in range(8):
                if i < 4:
                    figure_name = file_path + 'traffic_table_out_' + str(i) + "stage" + '.png'
                else:
                    figure_name = file_path + 'traffic_table_in_' + str(i - 4) + "stage" + '.png'
                self.draw(traffic_table_base_gpu_out_in_t[i], figure_name)

            figure_name = file_path + "traffic_table_out_sum.png"
            self.draw(traffic_table_base_gpu_out_in_t[0] + traffic_table_base_gpu_out_in_t[1] +
                      traffic_table_base_gpu_out_in_t[2] + traffic_table_base_gpu_out_in_t[3], figure_name)

            figure_name = file_path + "traffic_table_in_sum.png"
            self.draw(traffic_table_base_gpu_out_in_t[4] + traffic_table_base_gpu_out_in_t[5] +
                      traffic_table_base_gpu_out_in_t[6] + traffic_table_base_gpu_out_in_t[7], figure_name)

            time_end = time.time()
            print("%d nodes used. " % ((self.comm_size - 1) / 32 + 1))
            print("%.f seconds consumed." % (time_end - time_start))

            # self.check_traffic(2, traffic_table_base_gpu_out, traffic_table_base_gpu_in)
            # self.check_traffic(2, traffic_table_base_gpu_out)
        else:
            column_idx_to_process = self.allocate_idx_to_calculate()
            for gpu_out_idx in column_idx_to_process:
                traffic_table_base_gpu_out_tmp = np.zeros((self.N, 4))
                traffic_table_base_gpu_in_tmp = np.zeros((self.N, 4))
                # traffic_table_base_gpu_in_tmp = np.zeros((self.N, 3))
                # traffic_base_gpu_for_a_column_in = np.zeros((self.N,))
                time1 = time.time()
                # output_first_stage, output_second_stage = self.calculate_one_dcu_output_traffic(gpu_out_idx)

                traffic_table_base_gpu_out_tmp, traffic_table_base_gpu_in_tmp = self.calculate_4_dim_one_dcu_output_traffic(
                    gpu_out_idx)
                # traffic_table_base_gpu_in_tmp = self.calculate_3_dim_one_dcu_input_traffic(gpu_out_idx)
                time2 = time.time()
                traffic_table_base_gpu_out_in = np.concatenate(
                    (traffic_table_base_gpu_out_tmp, traffic_table_base_gpu_in_tmp), axis=1)
                self.comm.send(traffic_table_base_gpu_out_in, dest=self.master_rank)
                # self.comm.send(traffic_table_base_gpu_in_tmp, dest=self.master_rank)
                # self.comm.send(traffic_base_gpu_for_a_column_in, dest=self.master_rank)
                print('Col %d sent: %.4fs consumed.' % (gpu_out_idx, time2 - time1))

    def go(self):
        # self.compute_1_dim_traffic()
        self.compute_2_dim_traffic()
        # self.compute_3_dim_traffic()
        # self.compute_4_dim_traffic()

    def compute_2_dim_traffic_between_gpu_and_group_old(self, gpu_out_idx, gpu_in_idx):
        dcu_name = "cuda:" + str(self.rank % 4)
        device = torch.device(dcu_name if torch.cuda.is_available() else "cpu")

        traffic_gpu_to_gpu = 0

        for population_out_idx in self.map_table_split_old[gpu_out_idx].keys():
            traffic_src_to_dst = list()
            # gpu_in_idx_arr = self.get_idx_array(gpu_in_idx)
            # st = gpu_in_idx * self.n_gpu_per_group
            # en = st + self.n_gpu_per_group
            for in_idx in gpu_in_idx:
                for population_in_idx in self.map_table_split_old[int(in_idx)].keys():
                    tmp = int(population_in_idx * self.n + population_out_idx)
                    if tmp in self.conn_dict:
                        conn_number_estimate = self.neuron_number * self.size[population_in_idx] * \
                                               self.map_table_split_old[in_idx][population_in_idx] * self.degree[
                                                   population_in_idx] * self.conn_dict[tmp]
                        traffic_src_to_dst.append(conn_number_estimate)
                    else:
                        traffic_src_to_dst.append(0.0)

            sample_range = int(
                self.neuron_number * self.size[population_out_idx] * self.map_table_split_old[gpu_out_idx][
                    population_out_idx])
            sample_times = int(np.sum(traffic_src_to_dst))
            # print()
            # print("Sample range: %d, Sample times: %d" % (sample_range, sample_times))

            torch.cuda.empty_cache()
            n = 1
            if sample_range > 1e7 or sample_times > 1e8:
                n = 30
            traffic_src_to_gpu = self.sample(sample_range, sample_times, n_slice=n)
            torch.cuda.empty_cache()
            traffic_gpu_to_gpu += traffic_src_to_gpu

        # output.append(traffic_src_to_dcu)
        return traffic_gpu_to_gpu

    def compute_2_dim_traffic_between_two_gpu_old(self, gpu_out_idx, gpu_in_idx):
        dcu_name = "cuda:" + str(self.rank % 4)
        device = torch.device(dcu_name if torch.cuda.is_available() else "cpu")

        traffic_gpu_to_gpu = 0
        # print('before swap')
        # print(self.map_table_split[5861])
        #
        # print(self.map_table_split[4949])
        #
        # print('after swap')
        # print(self.map_table_split_old[5861])
        #
        # print(self.map_table_split_old[4949])
        for population_out_idx in self.map_table_split_old[gpu_out_idx].keys():
            traffic_src_to_dst = list()
            for population_in_idx in self.map_table_split_old[gpu_in_idx].keys():
                tmp = int(population_in_idx * self.n + population_out_idx)
                if tmp in self.conn_dict:
                    conn_number_estimate = self.neuron_number * self.size[population_in_idx] * \
                                           self.map_table_split_old[gpu_in_idx][population_in_idx] * self.degree[
                                               population_in_idx] * self.conn_dict[tmp]
                    traffic_src_to_dst.append(conn_number_estimate)
                else:
                    traffic_src_to_dst.append(0.0)

            sample_range = int(
                self.neuron_number * self.size[population_out_idx] * self.map_table_split_old[gpu_out_idx][
                    population_out_idx])
            sample_times = int(np.sum(traffic_src_to_dst))
            # print()
            # print("Sample range: %d, Sample times: %d" % (sample_range, sample_times))

            torch.cuda.empty_cache()
            n = 1
            if sample_range > 1e7 or sample_times > 1e8:
                n = 30
            traffic_src_to_gpu = self.sample(sample_range, sample_times, n_slice=n)
            torch.cuda.empty_cache()
            traffic_gpu_to_gpu += traffic_src_to_gpu

        # output.append(traffic_src_to_dcu)
        return traffic_gpu_to_gpu

    def is_changed_gpu_traffic(self, idx_list):
        for i in idx_list:
            for j in self.gpu_changed_idx:
                if int(i) == int(j):
                    return True
        return False

    def is_in_same_node(self, idx_list):
        if idx_list[0] // 4 == idx_list[1] // 4:
            return True
        return False

    def calculate_2_dim_output_traffic_for_map(self, idx):
        self.dimensions = 2
        tem_output_traffic = np.zeros((self.N, 1))
        route_dict_out_idx = self.get_route_dict_out(idx, range(self.N))

        for in_idx, in_idx_list in route_dict_out_idx.items():
            if len(in_idx_list) == 1:
                if self.is_changed_gpu_traffic([in_idx, idx]):
                    if not self.is_in_same_node([idx, in_idx]):
                        tmp1 = self.compute_2_dim_traffic_between_two_gpu(idx, in_idx)
                        tmp2 = self.compute_2_dim_traffic_between_two_gpu_old(idx, in_idx)
                        tem_output_traffic[idx] += (tmp1 - tmp2)
            else:
                in_idx_list_tmp = list(in_idx_list)[:]
                in_idx_list_tmp.append(idx)
                if self.is_changed_gpu_traffic(in_idx_list_tmp):
                    tmp1 = self.compute_2_dim_traffic_between_gpu_and_group(idx, in_idx_list)
                    tmp2 = self.compute_2_dim_traffic_between_gpu_and_group_old(idx, in_idx_list)
                    tem_output_traffic[idx] += (tmp1 - tmp2)

                route_dict_out_idx_tmp_1 = self.get_route_dict_out(in_idx, in_idx_list)
                for in_idx_1, in_idx_list_1 in route_dict_out_idx_tmp_1.items():
                    if self.is_changed_gpu_traffic([in_idx_1, idx]):
                        if not self.is_in_same_node([idx, in_idx_1]):
                            tmp1 = self.compute_2_dim_traffic_between_two_gpu(idx, in_idx_1)
                            tmp2 = self.compute_2_dim_traffic_between_two_gpu_old(idx, in_idx_1)
                            tem_output_traffic[int(in_idx)] += (tmp1 - tmp2)
        return tem_output_traffic

    def calculate_2_dim_traffic_for_map(self):
        if self.rank == self.master_rank:

            iter_count = 0
            iter_sum = 70
            max_traffic_iter = np.zeros((iter_sum,))
            average_traffic_iter = np.zeros((iter_sum,))
            min_traffic_iter = np.zeros((iter_sum,))
            while iter_count < iter_sum:
                time1 = time.time()
                if iter_count == 0:
                    self.traffic_table = np.load("traffic_table_base_dcu_out_in_2_dim20000_sequential.npy",
                                                 allow_pickle=True)
                    traffic_table_t = self.traffic_table.T
                    self.output_sum = traffic_table_t[0] + traffic_table_t[1] + traffic_table_t[2] + traffic_table_t[3]
                    self.traffic_table = self.output_sum

                    print("average: " + str(np.average(self.output_sum)))
                    print("max: " + str(np.max(self.output_sum)))
                    print("min: " + str(np.min(self.output_sum)))

                self.split_per_pop_by_step()
                # self.split_per_pop_by_step_v2()

                self.map_table_split = self.comm.bcast(self.map_table_split, self.master_rank)
                self.map_table_split_old = self.comm.bcast(self.map_table_split_old, self.master_rank)
                self.gpu_changed_idx = self.comm.bcast(self.gpu_changed_idx, self.master_rank)
                self.comm.barrier()
                time_start = time.time()
                # 这里是两个阶段，也可以不分阶段优化
                traffic_table_base_gpu_out = np.zeros((self.N, 1))
                # traffic_table_base_gpu_in = np.zeros((self.N, self.N))

                for col_idx in range(self.N):
                    msg_out = self.comm.recv(source=(col_idx % (self.comm_size - 1)))

                    traffic_table_base_gpu_out += msg_out
                traffic_table_base_gpu_out_t = traffic_table_base_gpu_out.T
                tmp_traffic = traffic_table_base_gpu_out_t[0]
                print("the div:  ------------")
                print("average: " + str(np.average(tmp_traffic)))
                print("max: " + str(np.max(tmp_traffic)) + " " + str(np.argmax(tmp_traffic)))
                print("min: " + str(np.min(tmp_traffic)) + " " + str(np.argmin(tmp_traffic)))
                self.output_sum = self.output_sum + tmp_traffic
                self.traffic_table = self.output_sum

                print("********")
                for i in self.gpu_changed_idx:
                    print(str(i) + " " + str(tmp_traffic[i]) + " " + str(self.output_sum[i]))
                print("********")
                max_traffic_iter[iter_count] = np.max(self.output_sum)
                min_traffic_iter[iter_count] = np.min(self.output_sum)
                average_traffic_iter[iter_count] = np.average(self.output_sum)
                print("the sum output:  ------------")
                print("average: " + str(np.average(self.output_sum)))
                print("max: " + str(np.max(self.output_sum)))
                print("min: " + str(np.min(self.output_sum)))
                time2 = time.time()
                print("one iteration consumed {0} seconds".format(time2 - time1))

                iter_count += 1
            np.save("map_table_split_v5.npy", self.map_table_split)
            print(max_traffic_iter)
            print(average_traffic_iter)
            print(min_traffic_iter)
            np.save("output_max.npy", max_traffic_iter)
            np.save("output_average.npy", average_traffic_iter)
            np.save("output_min.npy", min_traffic_iter)
        else:

            iter_count = 0
            while iter_count < 80:
                column_idx_to_process = self.allocate_idx_to_calculate()
                self.map_table_split = self.comm.bcast(self.map_table_split, self.master_rank)
                self.map_table_split_old = self.comm.bcast(self.map_table_split_old, self.master_rank)
                self.gpu_changed_idx = self.comm.bcast(self.gpu_changed_idx, self.master_rank)
                self.comm.barrier()
                for gpu_out_idx in column_idx_to_process:
                    traffic_table_base_gpu_out_tmp = np.zeros((self.N, 1))
                    time1 = time.time()
                    traffic_table_base_gpu_out_tmp = self.calculate_2_dim_output_traffic_for_map(gpu_out_idx)

                    time2 = time.time()
                    self.comm.send(traffic_table_base_gpu_out_tmp, dest=self.master_rank)
                    if gpu_out_idx % 1000 == 0:
                        print('Col %d sent: %.4fs consumed.' % (gpu_out_idx, time2 - time1))
                iter_count += 1

    def swap_pop(self):

        self.map_table_split_old = []
        for i in range(self.N):
            self.map_table_split_old.append(self.map_table_split[i].copy())
        list1 = np.array(list(self.map_table_split[self.gpu_idx[0]].keys()))
        list2 = np.array(list(self.map_table_split[self.gpu_idx[1]].keys()))

        # print(self.gpu_idx[0])
        # print("**********")
        # print(self.map_table_split[self.gpu_idx[0]])
        #
        # print(self.gpu_idx[1])
        # print("**********")
        # print(self.map_table_split[self.gpu_idx[1]])

        len = int(min(list1.shape[0], list2.shape[0]) / 2)
        for i in range(len):
            self.map_table_split[self.gpu_idx[0]][list2[i]] = self.map_table_split_old[self.gpu_idx[1]][list2[i]]
            self.map_table_split[self.gpu_idx[1]][list1[i]] = self.map_table_split_old[self.gpu_idx[0]][list1[i]]
            del self.map_table_split[self.gpu_idx[1]][list2[i]]
            del self.map_table_split[self.gpu_idx[0]][list1[i]]
        if self.rank == self.master_rank:
            print("swap {0} and {1}".format(self.gpu_idx[0], self.gpu_idx[1]))
            print("before swap")
            print(self.map_table_split_old[int(self.gpu_idx[0])])
            print(self.map_table_split_old[int(self.gpu_idx[1])])
            print("after swap")
            print(self.map_table_split[self.gpu_idx[0]])
            print(self.map_table_split[self.gpu_idx[1]])

    def split_pop_by_step(self, output_sum_output):
        self.map_table_split_old = np.empty((self.N,), dtype=dict)
        # Deep copy dictionary, old_split needs to be used in traffic calculation
        # don't to use shallow copy
        for i in range(self.N):
            self.map_table_split_old[i] = copy.deepcopy(self.map_table_split[i])
        self.step_sort_index = output_sum_output.argsort()
        self.gpu_changed_idx = []
        for i in range(self.step):
            # get the gpu that produces the most traffic
            max_traffic_gpu_idx = self.step_sort_index[output_sum_output.shape[0] - i - 1]
            # 拆分的本次的gpu的pop,
            max_pop_size = 0
            max_pop_idx = 0
            flag = 0
            for k, v in self.map_table_split[max_traffic_gpu_idx].items():
                if max_pop_size < self.size[k] and v == 1:
                    max_pop_size = self.size[k]
                    max_pop_idx = k
                if max_pop_size == 0:
                    continue
                flag = 1
            print(max_traffic_gpu_idx)
            print(self.map_table_split[max_traffic_gpu_idx])
            print(max_pop_idx)
            print(max_pop_size)
            if flag == 1:
                self.gpu_changed_idx.append(max_traffic_gpu_idx)
                start_gpu_idx = 4 * int(max_traffic_gpu_idx // 4)
                end_gpu_idx = start_gpu_idx + 4
                for idx in range(start_gpu_idx, end_gpu_idx):
                    self.map_table_split[idx][max_pop_idx] = 0.25
                if self.rank == self.master_rank:
                    print(start_gpu_idx)
                    print(max_traffic_gpu_idx)
                    print(end_gpu_idx)
                    print(self.map_table_split[max_traffic_gpu_idx][max_pop_idx])

        if self.rank == self.master_rank:
            for i in self.gpu_changed_idx:
                print("before swap")
                print(i)
                print(self.map_table_split_old[i])
                print("after swap")
                print(self.map_table_split[i])

    # 拆分产生最大流量的gpu的一半pop
    def split_per_pop_by_step(self, output_sum_output):
        self.map_table_split_old = np.empty((2000,), dtype=dict)
        for i in range(self.N):
            self.map_table_split_old[i] = copy.deepcopy(self.map_table_split[i])
        self.step_sort_index = output_sum_output.argsort()
        self.gpu_changed_idx = []
        for i in range(1, 10):
            max_traffic_gpu_idx = self.step_sort_index[output_sum_output.shape[0] - i]
            print(max_traffic_gpu_idx)
        idx = 0
        cnt = 0
        while cnt < self.step:
            # for i in range(self.step):
            max_traffic_gpu_idx = self.step_sort_index[output_sum_output.shape[0] - idx - 1]

            idx += 1
            # 拆分的本次的gpu的pop
            flag = 0
            max_gpu_pop_size = []
            for k, v in self.map_table_split[max_traffic_gpu_idx].items():
                if v == 1:
                    max_gpu_pop_size.append(self.size[k])
                    flag = 1
            if len(max_gpu_pop_size) == 0:
                print("max_traffic_gpu_idx " + str(max_traffic_gpu_idx))
                print(self.map_table_split[max_traffic_gpu_idx])
                print("traffic: " + str(output_sum_output[max_traffic_gpu_idx]))
                continue
            cnt += 1
            max_gpu_pop_size = np.array(max_gpu_pop_size)
            max_gpu_pop_size = np.sort(max_gpu_pop_size, axis=0)
            middle_size = max_gpu_pop_size[max_gpu_pop_size.shape[0] // 2]

            if flag == 1:
                pop_idx_list = []
                for k, v in self.map_table_split[max_traffic_gpu_idx].items():
                    if v == 1:
                        if self.size[k] >= middle_size:
                            pop_idx_list.append(k)
                self.gpu_changed_idx.append(max_traffic_gpu_idx)

                start_gpu_idx = 4 * int(max_traffic_gpu_idx // 4)
                end_gpu_idx = start_gpu_idx + 4
                for changed_idx in range(start_gpu_idx, end_gpu_idx):
                    self.gpu_changed_idx.append(changed_idx)
                    for k in pop_idx_list:
                        self.map_table_split[changed_idx][k] = 0.25

                if self.rank == self.master_rank:
                    print(start_gpu_idx)
                    print(max_traffic_gpu_idx)
                    print(end_gpu_idx)
                    print("before split")
                    print(self.map_table_split_old[max_traffic_gpu_idx])
                    print("after split")
                    print(self.map_table_split[max_traffic_gpu_idx])

    def split_per_pop_by_step_v2(self):
        '''
        用于分割pop，和split_per_pop_by_step相比，当一个pop已经被分割为0.25时，仍然会对其进行拆分
        这时size按照size*拆分比例 排序
        :return:
        '''
        self.map_table_split_old = np.empty((2000,), dtype=dict)
        for i in range(self.N):
            self.map_table_split_old[i] = copy.deepcopy(self.map_table_split[i])
        self.step_sort_index = self.output_sum.argsort()
        self.gpu_changed_idx = []

        cnt = 0
        while cnt < self.step:
            # for i in range(self.step):
            max_traffic_gpu_idx = self.step_sort_index[self.output_sum.shape[0] - cnt - 1]
            # 拆分的本次的gpu的pop
            max_gpu_pop_size = []
            for k, v in self.map_table_split[max_traffic_gpu_idx].items():
                max_gpu_pop_size.append(self.size[k] * v)

            cnt += 1
            max_gpu_pop_size = np.array(max_gpu_pop_size)
            max_gpu_pop_size = np.sort(max_gpu_pop_size, axis=0)
            middle_size = max_gpu_pop_size[max_gpu_pop_size.shape[0] // 2]

            start_gpu_idx = 4 * int(max_traffic_gpu_idx // 4)
            end_gpu_idx = start_gpu_idx + 4

            pop_idx_dict = {}

            for k, v in self.map_table_split[max_traffic_gpu_idx].items():
                if self.size[k] * v >= middle_size:
                    pop_idx_dict[k] = v

            for idx in range(start_gpu_idx, end_gpu_idx):
                self.gpu_changed_idx.append(idx)
                for k, v in pop_idx_dict.items():
                    if idx == max_traffic_gpu_idx:
                        self.map_table_split[idx][k] = v / 4
                    elif k in self.map_table_split[idx]:
                        self.map_table_split[idx][k] += v / 4
                    else:
                        self.map_table_split[idx][k] = v / 4

            # for idx in range(start_gpu_idx, max_traffic_gpu_idx):
            #     self.gpu_changed_idx.append(idx)
            #     for k, v in self.map_table_split[max_traffic_gpu_idx].items():
            #         if self.size[k] * v >= middle_size:
            #             tmp = float(v)
            #             if idx == max_traffic_gpu_idx:
            #                 self.map_table_split[idx][k] = tmp / 4
            #             else:
            #                 if k in self.map_table_split[idx]:
            #                     self.map_table_split[idx][k] += tmp / 4
            #                 else:
            #                     self.map_table_split[idx][k] = tmp / 4
            #
            # for idx in range(max_traffic_gpu_idx, end_gpu_idx):
            #     self.gpu_changed_idx.append(idx)
            #     for k, v in self.map_table_split[max_traffic_gpu_idx].items():
            #         if self.size[k] >= middle_size:
            #             tmp = float(v)
            #             if idx == max_traffic_gpu_idx:
            #                 self.map_table_split[idx][k] = tmp / 4
            #             else:
            #                 if k in self.map_table_split[idx]:
            #                     self.map_table_split[idx][k] += tmp
            #                 else:
            #                     self.map_table_split[idx][k] = tmp

            if self.rank == self.master_rank:
                print(start_gpu_idx)
                print(max_traffic_gpu_idx)
                print(end_gpu_idx)
                print("before split")
                # for i in range(start_gpu_idx, end_gpu_idx):
                #     print(self.map_table_split_old[i])
                # #
                for i in range(start_gpu_idx, end_gpu_idx):
                    print(self.map_table_split_old[i])
                # print(self.map_table_split_old[max_traffic_gpu_idx])
                #
                print("after split")
                for i in range(start_gpu_idx, end_gpu_idx):
                    print(self.map_table_split[i])
                #
                # for i in range(start_gpu_idx, end_gpu_idx):
                #     print(self.map_table_split[i])
                # print("test")
                # for i in self.gpu_changed_idx:
                #     start_idx = (i // 4) * 4
                #     end_idx = start_idx + 4
                #
                #     for k, v in self.map_table_split[i].items():
                #         sum_split = 0
                #         for j in range(start_idx, end_idx):
                #             if k in self.map_table_split[j]:
                #                 sum_split += self.map_table_split[j][k]
                #         if sum_split != 1:
                #             print(i)
                #             print(k)
                #             print(v)
                # print("all gpu is ")


def generate_map(self):
    iter_count = 0
    while iter_count < 10:

        time1 = time.time()
        if iter_count == 0:
            self.traffic_table = np.load("traffic_table_base_dcu_out_in_2_dimmap_10000_v1_cortical_v2.npy",
                                         allow_pickle=True)
            traffic_table_t = self.traffic_table.T
            self.output_sum = traffic_table_t[0] + traffic_table_t[1]
            self.traffic_table = self.output_sum
        if self.rank == self.master_rank:
            print("average: " + str(np.average(self.output_sum)))
            print("max: " + str(np.max(self.output_sum)))
            print("min: " + str(np.min(self.output_sum)))

        # self.split_pop_by_step()
        self.split_per_pop_by_step()
        # self.map_table_split = self.comm.bcast(self.map_table_split, self.master_rank)
        # self.map_table_split_old = self.comm.bcast(self.map_table_split_old, self.master_rank)
        # self.swap_pop()

        # self.gpu_idx = self.comm.bcast(self.gpu_idx, self.master_rank)
        # if self.rank == self.master_rank:
        #     print("bcast has been finished")
        # self.comm.barrier()
        # 计算新的流量
        self.calculate_2_dim_traffic_for_map()
        if self.rank == self.master_rank:
            print("one iteration has been finished")
            time2 = time.time()
            print("this iteration consumed {0} seconds".format((time2 - time1)))
        iter_count += 1

    if self.rank == self.master_rank:
        np.save("output_sum_result.npy", self.output_sum)


def test(self):
    iter_count = 0
    if iter_count == 0:
        self.traffic_table = np.load("traffic_table_base_dcu_out_in_2_dimmap_10000_v1_cortical_v2.npy",
                                     allow_pickle=True)
        traffic_table_t = self.traffic_table.T
        self.output_sum = traffic_table_t[0] + traffic_table_t[1]
        self.traffic_table = self.output_sum
    if self.rank == self.master_rank:
        print("max traffic gpu" + str(np.argmax(self.output_sum)))

        print("min traffic gpu" + str(np.argmin(self.output_sum)))
        print("average: " + str(np.average(self.output_sum)))
        print("max: " + str(np.max(self.output_sum)))
        print("min: " + str(np.min(self.output_sum)))

    self.split_pop_test()

    # self.map_table_split = self.comm.bcast(self.map_table_split, self.master_rank)
    # self.map_table_split_old = self.comm.bcast(self.map_table_split_old, self.master_rank)
    # self.swap_pop()

    # self.gpu_idx = self.comm.bcast(self.gpu_idx, self.master_rank)
    # if self.rank == self.master_rank:
    #     print("bcast has been finished")
    # self.comm.barrier()

    # 计算新的流量
    self.calculate_2_dim_traffic_for_map()
    if self.rank == self.master_rank:
        print("after one iteration")
        print("max traffic gpu" + str(np.argmax(self.output_sum)))

        print("min traffic gpu" + str(np.argmin(self.output_sum)))
        print("average: " + str(np.average(self.output_sum)))
        print("max: " + str(np.max(self.output_sum)))
        print("min: " + str(np.min(self.output_sum)))
        print("one iteration has been finished")


def calculate_2_dim_input_output_traffic_for_two_pop(self, idx):
    self.dimensions = 2
    tem_output_traffic = np.zeros((self.N, self.dimensions))
    route_dict_out_idx = self.get_route_dict_out(idx, range(self.N))
    for in_idx, in_idx_list in route_dict_out_idx.items():
        stage = 0
        if len(in_idx_list) == 1:
            if idx == 5710:
                tmp1 = self.compute_2_dim_traffic_between_two_gpu(idx, in_idx)
                tmp2 = self.compute_2_dim_traffic_between_two_gpu_old(idx, in_idx)
                tem_output_traffic[idx][stage] += (tmp1 - tmp2)
                # tem_input_traffic[in_idx][stage] += tmp
        else:
            if idx == 5710:
                tmp1 = self.compute_2_dim_traffic_between_gpu_and_group(idx, in_idx_list)
                tmp2 = self.compute_2_dim_traffic_between_gpu_and_group_old(idx, in_idx_list)
                tem_output_traffic[idx][stage] += (tmp1 - tmp2)
            # tem_input_traffic[in_idx][stage] += tmp
            route_dict_out_idx_tmp_1 = self.get_route_dict_out(in_idx, in_idx_list)
            for in_idx_1, in_idx_list_1 in route_dict_out_idx_tmp_1.items():
                stage = 1
                if in_idx == 5710:
                    tmp1 = self.compute_2_dim_traffic_between_two_gpu(idx, in_idx_1)
                    tmp2 = self.compute_2_dim_traffic_between_two_gpu_old(idx, in_idx_1)
                    tem_output_traffic[int(in_idx)][stage] += (tmp1 - tmp2)
                # tem_input_traffic[int(in_idx_1)][stage] += tmp
    return tem_output_traffic


def calculate_2_dim_traffic_for_two_pop(self):
    if self.rank == self.master_rank:
        time_start = time.time()
        # 这里是两个阶段，也可以不分阶段优化
        traffic_table_base_gpu_out = np.zeros((self.N, 2))
        self.traffic_table = np.load("traffic_table_base_dcu_out_in_2_dimmap_10000_v1_cortical_v2.npy",
                                     allow_pickle=True)
        traffic_table_t = self.traffic_table.T
        self.output_sum = traffic_table_t[0] + traffic_table_t[1]
        print("traffic: " + str(self.output_sum[5710]))
        # print("max traffic gpu" + str(np.argmax(self.output_sum)))
        #
        # print("min traffic gpu" + str(np.argmin(self.output_sum)))
        # print("average: " + str(np.average(self.output_sum)))
        # print("max: " + str(np.max(self.output_sum)))
        # print("min: " + str(np.min(self.output_sum)))

        # traffic_table_base_gpu_in = np.zeros((self.N, self.N))

        for col_idx in range(self.N):
            time1 = time.time()
            msg_out = self.comm.recv(source=(col_idx % (self.comm_size - 1)))

            traffic_table_base_gpu_out += msg_out
            time2 = time.time()
        print("calculate has been finished and consumed {0}".format(time2 - time1))
        traffic_table_base_gpu_out_t = traffic_table_base_gpu_out.T
        traffic_table_base_gpu_out_t_sum = traffic_table_base_gpu_out_t[0] + traffic_table_base_gpu_out_t[1]
        print(traffic_table_base_gpu_out_t_sum[5710])
        self.output_sum = traffic_table_base_gpu_out_t_sum + self.output_sum
        print("after calculate traffic: " + str(self.output_sum[5710]))
        # print("max traffic gpu" + str(np.argmax(self.output_sum)))
        #
        # print("min traffic gpu" + str(np.argmin(self.output_sum)))
        # print("average: " + str(np.average(self.output_sum)))
        # print("max: " + str(np.max(self.output_sum)))
        # print("min: " + str(np.min(self.output_sum)))

    else:
        column_idx_to_process = self.allocate_idx_to_calculate()
        for gpu_out_idx in column_idx_to_process:
            traffic_table_base_gpu_out_tmp = np.zeros((self.N, 2))
            time1 = time.time()
            traffic_table_base_gpu_out_tmp = self.calculate_2_dim_input_output_traffic_for_two_pop(gpu_out_idx)

            time2 = time.time()

            self.comm.send(traffic_table_base_gpu_out_tmp, dest=self.master_rank)


if __name__ == "__main__":
    # m = MapAnalysis()
    # m.generate_forwarding_table_17280(number_of_group=25, dcu_per_group=40, max_link=205, max_rate=1.085)

    Job = MapAnalysisParallel()
    Job.go()
    # Job.calculate_2_dim_traffic_for_two_pop()
    # import copy
    #
    # temp = copy.deepcopy(Job.map_table_split[9944])
    # Job.map_table_split[9944] = copy.deepcopy(Job.map_table_split[4949])
    # Job.map_table_split[4949] = copy.deepcopy(temp)
    # if Job.rank == Job.master_rank:
    #     print("before swap")
    #     print(Job.map_table_split[5710])
    #
    # Job.map_table_split_old = np.empty((10000,), dtype=dict)
    # for i in range(Job.N):
    #     Job.map_table_split_old[i] = copy.deepcopy(Job.map_table_split[i])
    #
    # start_gpu_idx = 4 * (5710 // 4)
    # end_gpu_idx = start_gpu_idx + 4
    #
    # for i in range(start_gpu_idx, end_gpu_idx):
    #     for k in Job.map_table_split[5710]:
    #         Job.map_table_split[i][k] = 0.25
    #
    # if Job.rank == Job.master_rank:
    #     print("after swap")
    #     print(Job.map_table_split[5710])
    # #
    # Job.calculate_2_dim_traffic_for_two_pop()
    #
    # Job.step = 5
    # Job.calculate_2_dim_traffic_for_map()
