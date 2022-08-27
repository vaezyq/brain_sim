"""
体素->dcu映射表的生成过程
"""
import copy
import numpy as np
import time
import os
import pickle
import matplotlib.pyplot as plt
import torch

from functions import Functions
from conn_analysis import ConnAnalysis
from parallelism import Parallelism

'''
generate_my_map()中交换体素的部分写得太丑，可以重构
'''


class GenerateMap(Functions, ConnAnalysis):
    def __init__(self):
        super().__init__()

        # map表地址
        self.map_root = self.root_path + 'tables/map_table/'
        self.map_table_path = self.map_root + self.map_version + '.pkl'
        self.map_version = "20000_sequential"
        if self.conn_version[0:5] == 'voxel':
            self.map_table_without_invalid_idx_path = self.map_table_path
        else:
            self.map_table_without_invalid_idx_path = self.map_version + '_without_invalid_idx.pkl'

        '''
        在按编号顺序分配的情况下，每个dcu中包含的体素/功能柱的个数
        N = 2000， n = 22603时，self.voxels_per_dcu = {11: 1397, 12: 603}
        N = 4000， n = 22603时，self.voxels_per_dcu = {5: 1397, 6: 2603}
        N = 17280，n = 171452时，self.voxels_per_dcu = {9: 1348, 10: 15932}
        '''
        self.voxels_per_dcu = {8: 8492, 9: 11508}
        self.N = 20000
        self.map_table_split_file = '20000_sequential_without_invalid_idx.pkl'

        self.sequential_map_without_invalid_index = np.load(self.map_table_split_file,
                                                            allow_pickle=True)

        # self.sequential_map_without_invalid_index = self.generate_sequential_map()  # 按编号顺序将体素分到dcu上
        # self.save_map_pkl(self.sequential_map_without_invalid_index, self.map_table_without_invalid_idx_path)
        # print(str(self.map_table_without_invalid_idx_path) + " saved")
        # if not os.path.exists(self.map_table_without_invalid_idx_path):
        #     self.save_map_pkl(self.sequential_map_without_invalid_index, self.map_table_without_invalid_idx_path)
        #     print(str(self.map_table_without_invalid_idx_path) + " saved")
        #
        # self.sequential_map = self.map_table_transfer(copy.deepcopy(self.sequential_map_without_invalid_index))
        # if not os.path.exists(self.map_table_path):
        #     self.save_map_pkl(self.sequential_map, self.map_table_path)
        #     print(str(self.map_table_path) + " saved")

    # 生成按编号顺序分组的map表

    def generate_sequential_map(self):
        map_table = list()
        for i in range(self.N):
            map_table.append([])

        cnt, idx = 0, 0
        for key in self.voxels_per_dcu:
            for i in range(self.voxels_per_dcu[key]):
                for j in range(key):
                    map_table[idx].append(cnt)
                    cnt += 1
                idx += 1
                print(idx)

        return map_table

    def generate_random_map(self):
        map_table = list()
        for i in range(self.N):
            map_table.append([])

        idxes = np.arange(0, self.n, dtype=int)
        np.random.shuffle(idxes)

        cnt, idx = 0, 0
        for key in self.voxels_per_dcu:
            for i in range(self.voxels_per_dcu[key]):
                for j in range(key):
                    map_table[idx].append(idxes[cnt])
                    cnt += 1
                idx += 1

        return map_table

    def initialize_sequential_map(self):
        map_table = list()
        for i in range(self.N):
            map_table.append([])

        big_p_idxes = list()
        idxes = np.arange(0, self.n, dtype=int)

        average = np.average(self.size_multi_degree)
        for i in range(168000, self.n):
            if self.size_multi_degree[i] > 0.4 * average:
                big_p_idxes.append(i)

        idxes = np.setdiff1d(idxes, big_p_idxes)
        interval = 168000 // len(big_p_idxes)
        for i in range(len(big_p_idxes)):
            idxes = np.insert(idxes, interval * i, big_p_idxes[i])

        cnt, idx = 0, 0
        for key in self.voxels_per_dcu:
            for i in range(self.voxels_per_dcu[key]):
                for j in range(key):
                    map_table[idx].append(idxes[cnt])
                    cnt += 1
                idx += 1

        return map_table

    # 生成按自己方案划分的group，按出入流量、体素size均分的原则把体素映射到dcu上
    def generate_my_map(self, iter_times=2000, show_iter_process=False, max_rate=1.15):
        path = self.traffic_table_root + "out_traffic_per_voxel_" + str(self.neuron_number) + ".npy"
        if os.path.exists(path):
            self.out_traffic, self.in_traffic = np.load(path), np.array([])
        else:
            self.out_traffic, self.in_traffic = self.cal_traffic_base_voxel()

        map_table = self.sequential_map_without_invalid_index.copy()

        origin_out, origin_in = self.sum_traffic_per_dcu(self.sequential_map_without_invalid_index)
        origin_size = list()
        for i in range(self.N):
            origin_size.append(np.sum(self.size[np.ix_(self.sequential_map_without_invalid_index[i])]))

        # 画图展示迭代过程，以及与迭代前相比的区别
        if show_iter_process:
            plt.ion()  # 开启interactive mode 成功的关键函数
            plt.figure(figsize=(19, 10))

        time1 = time.time()
        print('Begin to generate map...')
        # for i in range(iter_times):
        sum_out, sum_in = self.sum_traffic_per_dcu(map_table)
        size_per_group = np.array(origin_size)
        cnt = 0

        best_obj = 999

        while np.max(size_per_group) > np.average(size_per_group) * max_rate:
            # print("cnt=%d, average=%f, max/average=%f" %
            #       (cnt, np.average(size_per_group), np.max(size_per_group) / np.average(size_per_group)))
            best_obj = min(best_obj, np.max(size_per_group) / np.average(size_per_group))
            if cnt % 1000 == 0:
                print('best_obj: %.4f, target: %.4f' % (best_obj, max_rate))
            cnt += 1

            # out
            # 在前10大的组中随机挑一个
            # 找出流量最大的dcu与流量最小的dcu
            copy_sum_out = sum_out.copy()
            copy_sum_out.sort()
            idx_1 = np.random.randint(1, 10)
            idx_out_max = np.where(sum_out == copy_sum_out[-idx_1])[0][0]
            idx_out_min = np.argmin(sum_out)

            # 找出流量最大/最小的dcu所包含的功能柱各自的流量
            temp_max = self.out_traffic[np.ix_(map_table[idx_out_max])]
            temp_min = self.out_traffic[np.ix_(map_table[idx_out_min])]

            # 找随机第1-3大的功能柱
            copy_temp_max = temp_max.copy()
            copy_temp_max.sort()
            idx_2 = np.random.randint(1, 3)
            voxel_idx_out_max = np.where(temp_max == copy_temp_max[-idx_2])[0][0]
            voxel_idx_out_min = np.argmin(temp_min)

            max_voxel_overall_idx = map_table[idx_out_max][voxel_idx_out_max]
            min_voxel_overall_idx = map_table[idx_out_min][voxel_idx_out_min]

            temp = map_table[idx_out_max][voxel_idx_out_max]
            map_table[idx_out_max][voxel_idx_out_max] = map_table[idx_out_min][voxel_idx_out_min]
            map_table[idx_out_min][voxel_idx_out_min] = temp

            # 更新流量和
            sum_out[idx_out_max] = sum_out[idx_out_max] + temp_min[voxel_idx_out_min] - temp_max[voxel_idx_out_max]
            sum_out[idx_out_min] = sum_out[idx_out_min] - temp_min[voxel_idx_out_min] + temp_max[voxel_idx_out_max]

            # 更新size
            size_per_group[idx_out_max] = size_per_group[idx_out_max] + self.size[min_voxel_overall_idx] - self.size[
                max_voxel_overall_idx]
            size_per_group[idx_out_min] = size_per_group[idx_out_min] - self.size[min_voxel_overall_idx] + self.size[
                max_voxel_overall_idx]

            # in
            # # 找随机第1-10大的组
            # copy_sum_in = sum_in.copy()
            # copy_sum_in.sort()
            # idx = np.random.randint(1, 10)
            # idx_in_max = np.where(sum_in == copy_sum_in[-idx])[0][0]
            #
            # idx_in_min = np.argmin(sum_in)
            # temp_max = self.in_traffic[np.ix_(map_table[idx_in_max])]
            # temp_min = self.in_traffic[np.ix_(map_table[idx_in_min])]
            #
            # # 找随机第1-4大的体素
            # copy_temp_max = temp_max.copy()
            # copy_temp_max.sort()
            # idx = np.random.randint(1, 2)
            # voxel_idx_in_max = np.where(temp_max == copy_temp_max[-idx])[0][0]
            # voxel_idx_in_min = np.argmin(temp_min)
            #
            # temp = map_table[idx_in_max][voxel_idx_in_max]
            # map_table[idx_in_max][voxel_idx_in_max] = map_table[idx_in_min][voxel_idx_in_min]
            # map_table[idx_in_min][voxel_idx_in_min] = temp

            # size

            # size
            max_size_idx = np.argmax(size_per_group)
            min_size_idx = np.argmin(size_per_group)
            max_size_voxel_idx = np.argmax(self.size[np.ix_(map_table[max_size_idx])])
            min_size_voxel_idx = np.argmin(self.size[np.ix_(map_table[min_size_idx])])

            max_voxel_overall_idx = map_table[max_size_idx][max_size_voxel_idx]
            min_voxel_overall_idx = map_table[min_size_idx][min_size_voxel_idx]

            temp = map_table[max_size_idx][max_size_voxel_idx]
            map_table[max_size_idx][max_size_voxel_idx] = map_table[min_size_idx][min_size_voxel_idx]
            map_table[min_size_idx][min_size_voxel_idx] = temp

            # 更新size
            size_per_group[max_size_idx] = \
                size_per_group[max_size_idx] - self.size[max_voxel_overall_idx] + self.size[min_voxel_overall_idx]
            size_per_group[min_size_idx] = \
                size_per_group[min_size_idx] + self.size[max_voxel_overall_idx] - self.size[min_voxel_overall_idx]

            # 更新流量和
            traffic_max = self.out_traffic[max_voxel_overall_idx]
            traffic_min = self.out_traffic[min_voxel_overall_idx]
            sum_out[max_size_idx] = sum_out[max_size_idx] + traffic_min - traffic_max
            sum_out[min_size_idx] = sum_out[min_size_idx] - traffic_min + traffic_max

            # 画图查看迭代结果
            # if show_iter_process and i % 100 == 0:
            #     plt.clf()  # 清空之前画的
            #
            #     plt.subplot(211)
            #     plt.title('conn_number: average=%d, max=%d' % (np.average(sum_out), np.max(sum_out)))
            #     plt.xlabel('dcu')
            #     plt.ylabel('conn_number')
            #     plt.plot(origin_out, color='blue', linestyle='--', alpha=0.3)
            #     plt.plot(sum_out, color='blue')
            #
            #     plt.subplot(212)
            #     plt.title('size: average=%.6f, max=%.6f' % (np.average(size_per_group), np.max(size_per_group)))
            #     plt.xlabel('dcu')
            #     plt.ylabel('size')
            #     plt.plot(origin_size, linestyle='--', color='green', alpha=0.3)
            #     plt.plot(size_per_group, color='green')
            #
            #     plt.pause(0.05)

            # self.show_progress(i, iter_times, time1)

        time2 = time.time()
        print('Map generated. Iter times: %d, %.2fs consumed' % (iter_times, (time2 - time1)))

        best_obj = min(best_obj, np.max(size_per_group) / np.average(size_per_group))
        print('best_obj: %.4f, target: %.4f' % (best_obj, max_rate))

        # 展示每张dcu向外的流量之和
        sum_out, sum_in = self.sum_traffic_per_dcu(map_table)
        self.draw_traffic(origin_out, sum_out, name="traffic_out")

        # 展示每张dcu包含体素的size之和
        self.draw_traffic(origin_size, size_per_group, name="size")

        if self.map_table_without_invalid_idx_path is not None:
            self.save_map_pkl(map_table, self.map_table_without_invalid_idx_path)

        map_table = self.map_table_transfer(map_table)

        if self.map_table_path is not None:
            self.save_map_pkl(map_table, self.map_table_path)

        self.save_map_pkl(self.sequential_map_without_invalid_index, self.map_root + 'map_1200_sequential.pkl')

        return map_table

    # 把degree*size作为优化指标
    def generate_my_map_new(self, max_rate, max_iter_times, show_iter_process=False):
        self.show_basic_information()
        map_table = self.sequential_map_without_invalid_index.copy()

        origin_size_degree = self.cal_size_multi_degree(map_table)

        # 画图展示迭代过程，以及与迭代前相比的区别
        if show_iter_process:
            plt.ion()  # 开启interactive mode 成功的关键函数
            plt.figure(figsize=(10, 6), dpi=100)

        time1 = time.time()
        print('Begin to generate map...')
        size_per_dcu = np.array(origin_size_degree)
        cnt = 0

        best_obj = 999

        while np.max(size_per_dcu) > np.average(size_per_dcu) * max_rate and cnt < max_iter_times:
            # print("cnt=%d, average=%f, max/average=%f" %
            #       (cnt, np.average(size_per_dcu), np.max(size_per_dcu) / np.average(size_per_dcu)))
            best_obj = min(best_obj, np.max(size_per_dcu) / np.average(size_per_dcu))
            if cnt % 10000 == 0:
                print('iter %d: best_obj: %.8f, target: %.4f' % (cnt, best_obj, max_rate))
                # print('average = %.6f' % np.average(size_per_dcu))
            cnt += 1

            # size
            copy_size_per_dcu = size_per_dcu.copy()
            copy_size_per_dcu.sort()
            idx_temp1 = np.random.randint(1, 2)
            idx_temp2 = np.random.randint(1, 60)
            max_size_idx = np.where(size_per_dcu == copy_size_per_dcu[-idx_temp1])[0][0]
            min_size_idx = np.where(size_per_dcu == copy_size_per_dcu[idx_temp2 - 1])[0][0]

            temp1 = self.size_multi_degree[np.ix_(map_table[max_size_idx])].copy()
            temp2 = temp1.copy()
            temp2.sort()
            idx_temp = np.random.randint(1, 4)
            max_size_voxel_idx = np.where(temp1 == temp2[-idx_temp])[0][0]

            temp1 = self.size_multi_degree[np.ix_(map_table[min_size_idx])]
            temp2 = temp1.copy()
            temp2.sort()
            idx_temp = np.random.randint(1, 2)
            min_size_voxel_idx = np.where(temp1 == temp2[idx_temp - 1])[0][0]

            max_voxel_overall_idx = map_table[max_size_idx][max_size_voxel_idx]
            min_voxel_overall_idx = map_table[min_size_idx][min_size_voxel_idx]

            temp = map_table[max_size_idx][max_size_voxel_idx]
            map_table[max_size_idx][max_size_voxel_idx] = map_table[min_size_idx][min_size_voxel_idx]
            map_table[min_size_idx][min_size_voxel_idx] = temp

            # 更新size
            size_per_dcu[max_size_idx] = \
                size_per_dcu[max_size_idx] - self.size[max_voxel_overall_idx] * self.degree[
                    max_voxel_overall_idx] + self.size[min_voxel_overall_idx] * self.degree[min_voxel_overall_idx]
            size_per_dcu[min_size_idx] = \
                size_per_dcu[min_size_idx] + self.size[max_voxel_overall_idx] * self.degree[
                    max_voxel_overall_idx] - self.size[min_voxel_overall_idx] * self.degree[min_voxel_overall_idx]

            if show_iter_process and cnt % 1000 == 0:
                plt.clf()
                plt.title('iter: %d, size: average=%.6f, max / average =%.6f' %
                          (cnt, np.average(size_per_dcu), np.max(size_per_dcu) / np.average(size_per_dcu)))
                plt.xlabel('dcu')
                plt.ylabel('size')
                plt.plot(origin_size_degree, linestyle='--', color='green', alpha=0.3)
                plt.plot(size_per_dcu, color='green')

                plt.pause(0.5)

        time2 = time.time()
        print('Map generated. Iter times: %d, %.2fs consumed' % (cnt, (time2 - time1)))

        best_obj = min(best_obj, np.max(size_per_dcu) / np.average(size_per_dcu))
        print('best_obj: %.6f, target: %.6f' % (best_obj, max_rate))

        ultimate_size_degree = self.cal_size_multi_degree(map_table)
        print('Check Size Degree: %.6f' % (np.max(ultimate_size_degree) / np.average(ultimate_size_degree)))

        # 展示每张dcu包含体素的size之和
        self.draw_traffic(origin_size_degree, size_per_dcu, name="size_degree")

        if self.conn_version[0:5] != 'voxel':
            self.save_map_pkl(map_table, self.map_table_without_invalid_idx_path)

        map_table = self.map_table_transfer(map_table)

        if self.map_table_path is not None:
            self.save_map_pkl(map_table, self.map_table_path)

        return map_table

    def map_table_transfer(self, map_table):
        mp_171452_to_226030 = list()
        for i in range(len(self.origin_size)):
            if self.origin_size[i] != 0:
                mp_171452_to_226030.append(i)

        for i in range(len(map_table)):
            for j in range(len(map_table[i])):
                # print(map_table[i][j])
                map_table[i][j] = mp_171452_to_226030[map_table[i][j]]

        return map_table

    # NOTE(@lyh): added on 2022.3.9
    def map_table_transfer_split(self, N, map_table_split):
        mp_171508_to_227030 = list()
        for i in range(len(self.origin_size)):
            if self.origin_size[i] != 0:
                mp_171508_to_227030.append(i)

        new_map_table_split = list()

        for gpu_idx in range(N):
            keys = list(map_table_split[gpu_idx].keys())
            keys.sort()
            new_map = dict()
            for i in range(len(keys)):
                old_key = keys[i]
                new_key = mp_171508_to_227030[old_key]
                new_map[new_key] = map_table_split[gpu_idx][old_key]
            new_map_table_split.append(new_map)

        return new_map_table_split

    def cal_size_multi_degree(self, map_table):
        assert type(map_table[0]) == list
        size_degree = np.zeros(self.N)
        for i in range(self.N):
            for population_idx in map_table[i]:
                size_degree[i] += self.size[population_idx] * self.degree[population_idx]
        # print(np.max(size_degree) / np.average(size_degree))
        return size_degree

    def cal_size_multi_degree_split(self, map_table):
        assert type(map_table[0]) == dict
        size_degree = np.zeros(self.N)
        for gpu_idx in range(self.N):
            for population_idx in map_table[gpu_idx].keys():
                size_degree[gpu_idx] += map_table[gpu_idx][population_idx] * self.size[population_idx] * self.degree[
                    population_idx]

        return size_degree

    def draw_traffic(self, traffic_origin, traffic_now, name):
        plt.figure(figsize=(10, 6), dpi=150)
        plt.title(name + ': average=%f, max=%f' % (np.average(traffic_now), np.max(traffic_now)))
        plt.xlabel('dcu')
        plt.ylabel('traffic')

        # plt.xlim(0, self.N)
        # plt.ylim(1000, 4000)

        plt.plot(traffic_origin, color='blue', linestyle='--', alpha=0.3, label='before')
        plt.plot(traffic_now, color='blue', label='now')
        plt.legend(fontsize=13)
        figure_name = name + "_per_dcu_" + self.map_version + ".png"
        plt.show()
        plt.savefig(self.root_path + "tables/map_table/" + figure_name)
        print(self.root_path + "tables/map_table/" + figure_name + " saved.")

    # NOTE(@lyh): added on 15:45 2022.2.15
    # 将map_table由list格式转变为拆分population所需要的dict格式
    def map_table_normal_to_split(self, map_table):
        map_table_split = list()
        for i in range(self.N):
            map_table_split.append(dict())

        # 生成按标号顺序将标号顺序映射到GPU的映射表，此时每个体素的初始部分都为1
        for gpu_idx in range(self.N):
            for population_idx in map_table[str(gpu_idx)]:
                map_table_split[gpu_idx][population_idx] = 1

        return map_table_split

    # NOTE(@lyh): added on 2022.3.7
    # 将map_table由拆分population所需要的dict格式转变为list格式(仅限于没有真正将population拆分的)
    def map_table_split_to_normal(self, map_table):
        map_table_split = list()
        for i in range(self.N):
            map_table_split.append(list())

        # 生成按标号顺序将标号顺序映射到GPU的映射表，此时每个体素的初始部分都为1
        for gpu_idx in range(self.N):
            for population_idx in map_table[gpu_idx].keys():
                map_table_split[gpu_idx].append(population_idx)

        return map_table_split

    # NOTE(@lyh): added on 14:48 2022.2.16
    # 把指定的population分为k_split份，拆分到同一节点的其他进程中
    def adjust_map(self, map_table_split, k_traffic, k_split):
        if type(map_table_split[0]) != 'dict':
            map_table_split = self.map_table_normal_to_split(map_table_split)

        population_with_large_traffic_idxes = self.pick_population_idx_with_large_out_traffic(k_traffic)
        print(population_with_large_traffic_idxes)

        for gpu_idx in range(self.N):
            for population_idx in map_table_split[gpu_idx].keys():
                if population_idx in population_with_large_traffic_idxes:
                    population_with_large_traffic_idxes.remove(population_idx)
                    for idx in range(gpu_idx // 4 * 4, gpu_idx // 4 * 4 + k_split):
                        map_table_split[idx][population_idx] = 1 / k_split

        # self.save_map_pkl(map_table_split, self.map_table_without_invalid_idx_path)

        return map_table_split

    # NOTE(@lyh): added on 14:48 2022.2.23
    # 相比于adjust_map()函数，这个函数会把流量大的、位于同一节点的population分散开
    def adjust_map_disperse(self, map_table_split, k_traffic, k_split):
        if type(map_table_split[0]) != 'dict':
            map_table_split = self.map_table_normal_to_split(map_table_split)

        population_with_large_traffic_idxes = self.pick_population_idx_with_large_out_traffic(k_traffic)
        print("k_split =", k_split)
        print(population_with_large_traffic_idxes)

        mp_171452_to_226030 = list()
        for i in range(len(self.origin_size)):
            if self.origin_size[i] != 0:
                mp_171452_to_226030.append(i)

        population_with_large_traffic_absolute_idxes = list()
        for idx in population_with_large_traffic_idxes:
            population_with_large_traffic_absolute_idxes.append(mp_171452_to_226030[idx])
        print(population_with_large_traffic_absolute_idxes)

        used_gpu_idx = list()  # 已经有拆分了的population的GPU编号

        for gpu_idx in range(self.N):
            for population_idx in map_table_split[gpu_idx].keys():
                if population_idx in population_with_large_traffic_idxes and map_table_split[gpu_idx][
                    population_idx] == 1:
                    # 找到还没有拆分了的population的gpu编号（本应是节点编号，这里为了写程序方便）
                    curr_gpu_idx = gpu_idx
                    cnt_loop = 0
                    while curr_gpu_idx in used_gpu_idx and curr_gpu_idx >= 0:  # 这里有风险
                        curr_gpu_idx = (curr_gpu_idx + 4) % self.N
                        cnt_loop += 1
                        assert cnt_loop < self.N
                    # 将population拆分到对应节点的4个进程（gpu）上去
                    for idx in range(curr_gpu_idx // 4 * 4, curr_gpu_idx // 4 * 4 + k_split):
                        map_table_split[idx][population_idx] = (1 / k_split)
                        used_gpu_idx.append(idx)

        for p_idx in population_with_large_traffic_idxes:
            for gpu_idx in range(self.N):
                if p_idx in map_table_split[gpu_idx] and map_table_split[gpu_idx][p_idx] == 1:
                    map_table_split[gpu_idx].pop(p_idx)

        # self.save_map_pkl(map_table_split, self.map_table_without_invalid_idx_path)

        return map_table_split

    # NOTE(@lyh): added on 10:30 2022.2.21
    # 以贪心的方式，更新拆分population后的map_table
    def generate_map_split(self, max_rate, max_iter_times, k_traffic, k_split, show_iter_process=False):
        self.show_basic_information()
        # random_map = self.initialize_sequential_map()
        # random_map = self.generate_sequential_map()
        # map_table = self.map_table_normal_to_split(self.sequential_map_without_invalid_index)
        max_i = 0
        for i in range(20000):
            max_tmp = max(self.sequential_map_without_invalid_index[str(i)])
            max_i = max(max_i, max_tmp)
        print(max_i)
        map_table_split = self.map_table_normal_to_split(self.sequential_map_without_invalid_index)

        max_i = 0
        print("----------")
        for i in range(20000):
            max_tmp = max(map_table_split[i])
            max_i = max(max_i, max_tmp)
        print(max_i)
        # map_table = self.adjust_map_disperse(map_table_split, k_traffic=k_traffic, k_split=k_split)
        map_table = map_table_split

        origin_size_degree = self.cal_size_multi_degree_split(map_table)

        # 画图展示迭代过程，以及与迭代前相比的区别
        if show_iter_process:
            plt.ion()  # 开启interactive mode 成功的关键函数
            plt.figure(figsize=(10, 6), dpi=100)

        time1 = time.time()
        print('Begin to generate map...')
        # print("k_traffic = %d" % k_traffic)
        size_per_gpu = np.array(origin_size_degree)  # deep copy
        average = np.average(size_per_gpu)
        cnt = 0
        best_obj = 999

        while np.max(size_per_gpu) > np.average(size_per_gpu) * max_rate and cnt < max_iter_times:
            # print("cnt=%d, average=%f, max/average=%f" %
            #       (cnt, np.average(size_per_gpu), np.max(size_per_gpu) / np.average(size_per_gpu)))

            best_obj = min(best_obj, np.max(size_per_gpu) / average)
            if cnt % 100 == 0:
                print('iter %d: best_obj: %.8f, target: %.4f' % (cnt, best_obj, max_rate))
                # print('average = %.6f' % average)
            cnt += 1

            # 随机选出size * degree小/大的进程(GPU)，并返回其编号
            copy_size_per_dcu = size_per_gpu.copy()
            copy_size_per_dcu.sort()
            max_idx_temp = np.random.randint(1, 2)
            min_idx_temp = np.random.randint(1, 60)
            max_size_gpu_idx = np.where(size_per_gpu == copy_size_per_dcu[-max_idx_temp])[0][0]
            min_size_gpu_idx = np.where(size_per_gpu == copy_size_per_dcu[min_idx_temp - 1])[0][0]

            # 随机选出流量大的population，并得到其绝对编号，对于已经拆分的population不去选择
            size_per_population_inside_gpu = dict()
            for population_idx in map_table[max_size_gpu_idx].keys():
                if map_table[max_size_gpu_idx][population_idx] == 1:  # population未被拆分
                    size_per_population_inside_gpu[population_idx] = self.size_multi_degree[population_idx] * \
                                                                     map_table[max_size_gpu_idx][population_idx]
                else:  # population在预处理阶段已经被拆分
                    size_per_population_inside_gpu[population_idx] = 0  # 对于已经拆分的population记为0，以确保不会被选择
            temp2 = sorted(size_per_population_inside_gpu.items(), key=lambda x: x[1], reverse=True)
            idx_temp = np.random.randint(0, 3)
            max_voxel_absolute_idx = temp2[idx_temp][0]

            # 随机选出流量小的population，并得到其绝对编号，对于已经拆分的population不去选择
            size_per_population_inside_gpu = dict()
            for population_idx in map_table[min_size_gpu_idx].keys():  # population未被拆分
                if map_table[min_size_gpu_idx][population_idx] == 1:
                    size_per_population_inside_gpu[population_idx] = self.size_multi_degree[population_idx] * \
                                                                     map_table[min_size_gpu_idx][population_idx]
                else:  # population在预处理阶段已经被拆分
                    size_per_population_inside_gpu[population_idx] = 1e3  # 对于已经拆分的population记为很大的数，以确保不会被选择

            temp2 = sorted(size_per_population_inside_gpu.items(), key=lambda x: x[1], reverse=False)
            idx_temp = np.random.randint(0, 1)
            min_voxel_absolute_idx = temp2[idx_temp][0]

            # 选出来的大流量population与小流量population互换位置
            max_key, min_key = max_voxel_absolute_idx, min_voxel_absolute_idx
            map_table[max_size_gpu_idx][min_key] = map_table[max_size_gpu_idx].pop(max_key)
            map_table[min_size_gpu_idx][max_key] = map_table[min_size_gpu_idx].pop(min_key)

            # 更新size * degree
            size_per_gpu[max_size_gpu_idx] = size_per_gpu[max_size_gpu_idx] - self.size_multi_degree[
                max_voxel_absolute_idx] + self.size_multi_degree[min_voxel_absolute_idx]
            size_per_gpu[min_size_gpu_idx] = size_per_gpu[min_size_gpu_idx] + self.size_multi_degree[
                max_voxel_absolute_idx] - self.size_multi_degree[min_voxel_absolute_idx]

            if show_iter_process and cnt % 10000 == 0:
                plt.clf()
                plt.title('iter: %d, size: average=%.6f, max / average =%.6f' %
                          (cnt, np.average(size_per_gpu), np.max(size_per_gpu) / np.average(size_per_gpu)))
                plt.xlabel('GPU index')
                plt.ylabel('Size * degree')
                plt.plot(origin_size_degree, linestyle='--', color='green', alpha=0.3)
                plt.plot(size_per_gpu, color='green')

                plt.pause(0.5)

        time2 = time.time()
        print('Map generated. Iter times: %d, %.2fs consumed' % (cnt, (time2 - time1)))

        best_obj = min(best_obj, np.max(size_per_gpu) / np.average(size_per_gpu))
        print('best_obj: %.6f, target: %.6f' % (best_obj, max_rate))

        ultimate_size_degree = self.cal_size_multi_degree_split(map_table)
        print('Check Size Degree: %.6f' % (np.max(ultimate_size_degree) / np.average(ultimate_size_degree)))

        # 展示每张dcu包含体素的size之和
        self.draw_traffic(origin_size_degree, size_per_gpu, name="size_degree")

        if self.conn_version[0:5] != 'voxel':
            self.save_map_pkl(map_table, self.map_table_without_invalid_idx_path)

        # map_table = self.map_table_transfer_split(self.N, map_table)

        if self.map_table_path is not None:
            self.save_map_pkl(map_table, self.map_table_path)

    # 以单个population的流量和为目标更新map_table(流量为简单加和，没有并包)
    def generate_map_base_population_out_traffic(self, max_rate, max_iter_times, show_iter_process=False):
        map_table = self.map_table_normal_to_split(self.sequential_map_without_invalid_index)
        map_table = self.adjust_map_disperse(map_table.copy(), k_traffic=30, k_split=2)

        out_traffic_per_gpu = np.zeros(self.N)
        for gpu_idx in range(self.N):
            for p_idx in map_table[gpu_idx].keys():
                out_traffic_per_gpu[gpu_idx] += self.out_traffic_per_population[p_idx] * map_table[gpu_idx][p_idx]

        origin_out_traffic_per_gpu = out_traffic_per_gpu.copy()

        # 画图展示迭代过程，以及与迭代前相比的区别
        if show_iter_process:
            plt.ion()  # 开启interactive mode 成功的关键函数
            plt.figure(figsize=(10, 6), dpi=100)

        time1 = time.time()
        print('Begin to generate map...')
        average = np.average(out_traffic_per_gpu)
        cnt = 0
        best_obj = 999

        while np.max(out_traffic_per_gpu) > np.average(out_traffic_per_gpu) * max_rate and cnt < max_iter_times:
            # print("cnt=%d, average=%f, max/average=%f" %
            #       (cnt, np.average(size_per_gpu), np.max(size_per_gpu) / np.average(size_per_gpu)))

            best_obj = min(best_obj, np.max(out_traffic_per_gpu) / average)
            if cnt % 10000 == 0:
                print('iter %d: best_obj: %.8f, target: %.4f' % (cnt, best_obj, max_rate))
                # print('average = %.6f' % average)
            cnt += 1

            # 随机选出size * degree小/大的进程(GPU)，并返回其编号
            copy_traffic_per_dcu = out_traffic_per_gpu.copy()
            copy_traffic_per_dcu.sort()
            max_idx_temp = np.random.randint(1, 5)
            min_idx_temp = np.random.randint(1, 60)

            max_traffic_gpu_idx = np.where(out_traffic_per_gpu == copy_traffic_per_dcu[-max_idx_temp])[0][0]
            min_traffic_gpu_idx = np.where(out_traffic_per_gpu == copy_traffic_per_dcu[min_idx_temp - 1])[0][0]

            # 随机选出流量大的population，并得到其绝对编号，对于已经拆分的population不去选择
            traffic_per_population_inside_gpu = dict()
            for population_idx in map_table[max_traffic_gpu_idx].keys():
                if map_table[max_traffic_gpu_idx][population_idx] == 1:  # population未被拆分
                    traffic_per_population_inside_gpu[population_idx] = self.size_multi_degree[population_idx] * \
                                                                        map_table[max_traffic_gpu_idx][population_idx]
                else:  # population在预处理阶段已经被拆分
                    traffic_per_population_inside_gpu[population_idx] = 0  # 对于已经拆分的population记为0，以确保不会被选择
            temp2 = sorted(traffic_per_population_inside_gpu.items(), key=lambda x: x[1], reverse=True)
            idx_temp = np.random.randint(0, 3)
            max_voxel_absolute_idx = temp2[idx_temp][0]

            # 随机选出流量小的population，并得到其绝对编号，对于已经拆分的population不去选择
            traffic_per_population_inside_gpu = dict()
            for population_idx in map_table[min_traffic_gpu_idx].keys():  # population未被拆分
                if map_table[min_traffic_gpu_idx][population_idx] == 1:
                    traffic_per_population_inside_gpu[population_idx] = self.size_multi_degree[population_idx] * \
                                                                        map_table[min_traffic_gpu_idx][population_idx]
                else:  # population在预处理阶段已经被拆分
                    traffic_per_population_inside_gpu[population_idx] = 1e3  # 对于已经拆分的population记为很大的数，以确保不会被选择

            temp2 = sorted(traffic_per_population_inside_gpu.items(), key=lambda x: x[1], reverse=False)
            idx_temp = np.random.randint(0, 1)
            min_voxel_absolute_idx = temp2[idx_temp][0]

            # 选出来的大流量population与小流量population互换位置
            max_key, min_key = max_voxel_absolute_idx, min_voxel_absolute_idx
            map_table[max_traffic_gpu_idx][min_key] = map_table[max_traffic_gpu_idx].pop(max_key)
            map_table[min_traffic_gpu_idx][max_key] = map_table[min_traffic_gpu_idx].pop(min_key)

            # 更新size * degree
            out_traffic_per_gpu[max_traffic_gpu_idx] = out_traffic_per_gpu[max_traffic_gpu_idx] - \
                                                       self.out_traffic_per_population[
                                                           max_voxel_absolute_idx] + self.out_traffic_per_population[
                                                           min_voxel_absolute_idx]
            out_traffic_per_gpu[min_traffic_gpu_idx] = out_traffic_per_gpu[min_traffic_gpu_idx] + \
                                                       self.out_traffic_per_population[
                                                           max_voxel_absolute_idx] - self.out_traffic_per_population[
                                                           min_voxel_absolute_idx]

            if show_iter_process and cnt % 10000 == 0:
                plt.clf()
                plt.title('iter: %d, size: average=%.6f, max / average =%.6f' %
                          (cnt, np.average(out_traffic_per_gpu),
                           np.max(out_traffic_per_gpu) / np.average(out_traffic_per_gpu)))
                plt.xlabel('GPU index')
                plt.ylabel('Size * degree')
                plt.plot(origin_out_traffic_per_gpu, linestyle='--', color='green', alpha=0.3)
                plt.plot(out_traffic_per_gpu, color='green')

                plt.pause(0.5)

        time2 = time.time()
        print('Map generated. Iter times: %d, %.2fs consumed' % (cnt, (time2 - time1)))

        best_obj = min(best_obj, np.max(out_traffic_per_gpu) / np.average(out_traffic_per_gpu))
        print('best_obj: %.6f, target: %.6f' % (best_obj, max_rate))

        ultimate_size_degree = self.cal_size_multi_degree_split(map_table)
        print('Check Size Degree: %.6f' % (np.max(ultimate_size_degree) / np.average(ultimate_size_degree)))

        # 展示每张dcu包含体素的size之和
        self.draw_traffic(origin_out_traffic_per_gpu, out_traffic_per_gpu, name="size_degree")

        if self.conn_version[0:5] != 'voxel':
            self.save_map_pkl(map_table, self.map_table_without_invalid_idx_path)

        pass

    def test(self):
        path = "../tables/map_table/map_10000_split_14_1.pkl"
        new_path = "../tables/map_table/map_10000_split_14_1.pkl"
        map_table_split = self.read_map_pkl(path)
        # new_map = self.map_table_split_to_normal(map_table_split)
        new_map = self.map_table_transfer_split(10000, map_table_split)
        self.save_map_pkl(new_map, new_path)


class GenerateMapParallelMaster(Parallelism, GenerateMap):
    def __init__(self):
        super().__init__()

        self.base_map_version = None
        self.new_map_version = None
        self.figure_save_path = None

        self.traffic_base_gpu_path = None
        self.traffic_base_population_path = None

        self.map_table = None
        self.traffic_base_gpu = None
        self.traffic_base_population = None
        self.origin_out, self.origin_in = None, None
        self.traffic_out_per_gpu, self.traffic_in_per_gpu = None, None

    def initialize_master(self):
        time1 = time.time()
        print("Begin to initialize master rank...")

        # self.traffic_base_gpu_path = self.traffic_table_root + "traffic_table_base_dcu_" + self.base_map_version + ".npy"
        self.traffic_base_gpu_path = 'traffic_table_base_dcu_out_in_2_dimmap_10000_v1_cortical_v2.npy'
        self.traffic_base_population_path = '/public/home/ssct005t/lyh_route/tables/traffic_table/traffic_base_cortical_map_10000_sequential_cortical_v2.pkl'

        # self.traffic_base_population_path = self.traffic_table_root + "traffic_base_cortical_" + self.base_map_version + ".pkl"

        self.traffic_base_gpu = np.load(self.traffic_base_gpu_path)
        print(self.traffic_base_gpu_path + " loaded.")

        with open(self.traffic_base_population_path, 'rb') as f:
            self.traffic_base_population = pickle.load(f)
        print(self.traffic_base_population_path + ' loaded.')

        self.traffic_out_per_gpu, self.traffic_in_per_gpu = np.empty(self.N), np.empty(self.N)
        self.cal_traffic()
        self.origin_out, self.origin_in = self.traffic_out_per_gpu, self.traffic_in_per_gpu
        np.save(self.figure_save_path + "origin_out.npy", self.origin_out)
        np.save(self.figure_save_path + "origin_in.npy", self.origin_in)

        time2 = time.time()
        print('Initialization complete. %.2fs consumed' % (time2 - time1))

    # could be further accelerated
    def cal_traffic(self):
        traffic_base_gpu_t = self.traffic_base_gpu.T
        self.traffic_out_per_gpu = traffic_base_gpu_t[0] + traffic_base_gpu_t[1]
        self.traffic_in_per_gpu = traffic_base_gpu_t[2] + traffic_base_gpu_t[3]

        # for i in range(self.N):
        #     inside_node_idx = np.arange(i // 4 * 4, i // 4 * 4 + 4)
        #     self.traffic_out_per_gpu[i] = np.sum(self.traffic_base_gpu[:, i])
        #     self.traffic_in_per_gpu[i] = np.sum(self.traffic_base_gpu[i, :])
        #
        #     for idx in inside_node_idx:
        #         self.traffic_out_per_gpu[i] -= self.traffic_base_gpu[idx][i]
        #         self.traffic_in_per_gpu[i] -= self.traffic_base_gpu[i][idx]

        return

    @staticmethod
    def random_selection_max(elements_to_choose, k):
        """
        :param elements_to_choose: Elements to be selected
        :param k: Pick one of the top k elements randomly
        :return: the index of the selected element
        """

        k = min(k, len(elements_to_choose))

        if k == 1:
            idx = np.argmax(elements_to_choose)
        else:
            if type(elements_to_choose) is list:
                array = np.array(elements_to_choose)
            else:
                array = elements_to_choose.copy()
            array.sort()
            random_n = np.random.randint(1, k + 1)
            idx = np.where(elements_to_choose == array[-random_n])[0][0]

        return idx

    @staticmethod
    def random_selection_min(elements_to_choose, k):
        """
        :param elements_to_choose: Elements to be selected
        :param k: Pick one of the first k elements randomly
        :return: the index of the selected element
        """
        k = min(k, len(elements_to_choose))

        if k == 1:
            idx = np.argmin(elements_to_choose)
        else:
            if type(elements_to_choose) is list:
                array = np.array(elements_to_choose)
            else:
                array = elements_to_choose.copy()
            array.sort()
            random_n = np.random.randint(0, k)
            idx = np.where(elements_to_choose == array[random_n])[0][0]

        return idx

    def find_cortical_with_large_traffic_out(self, max_i, max_j, max_k=1):
        """
        max_i, max_j is better to be 1. If they are more than 1, it will cause copying and sorting of a array
        of size self.N, thus leading to a considerable consumption of time.
        """

        # old method(before 2021.12.07)
        # src_gpu_idx = self.random_selection_max(self.traffic_out_per_dcu, max_i)
        # print("largest traffic:", src_gpu_idx, self.traffic_out_per_dcu[src_gpu_idx])
        # x = src_gpu_idx
        # while x == src_gpu_idx:
        #     x = self.random_selection_max(self.traffic_base_dcu[:, src_gpu_idx], max_j)
        # p_abs_idx = self.random_selection_max(self.traffic_base_cortical[x][src_gpu_idx], max_k)
        # print("large out traffic cortical:", x, src_gpu_idx, p_abs_idx, self.size[self.map_table[src_gpu_idx][p_abs_idx]])

        # new method
        src_gpu_idx = self.random_selection_max(self.traffic_out_per_gpu, max_i)
        x = -1

        population_idxes = list(self.map_table[src_gpu_idx].keys()).copy()
        population_idxes = np.array(population_idxes)
        traffic_sent_per_population = np.zeros(len(population_idxes))
        for dst in range(self.N):
            for p_rel_idx in range(len(population_idxes)):
                traffic_sent_per_population[p_rel_idx] += self.traffic_base_population[dst][src_gpu_idx][p_rel_idx]

        # 排序并选出流量大且未被拆分的population
        sort_idx = np.argsort(traffic_sent_per_population)
        idx = np.random.randint(-max_j, 0)
        p_abs_idx = population_idxes[sort_idx[idx]]
        while self.map_table[src_gpu_idx][p_abs_idx] != 1:
            idx -= 1
            p_abs_idx = population_idxes[sort_idx[idx]]

        print("large out traffic cortical:", src_gpu_idx, p_abs_idx)
        print("largest traffic: %.4e" % traffic_sent_per_population[np.where(population_idxes == p_abs_idx)[0][0]])

        return {'src': src_gpu_idx, 'dst': x, 'p_idx': p_abs_idx}

    def find_cortical_with_small_traffic_out(self, max_i=1, max_j=1, max_k=1):
        # old method(before 2021.12.07)
        # y = self.random_selection_min(self.traffic_out_per_dcu, max_i)
        # x = self.random_selection_min(self.traffic_base_dcu[:, y], max_j)
        # z = self.random_selection_min(self.traffic_base_cortical[x][y], max_k)
        # print("small out traffic cortical:", x, y, z)

        # new method
        src_gpu_idx = self.random_selection_min(self.traffic_out_per_gpu, max_i)
        x = -1

        population_idxes = list(self.map_table[src_gpu_idx].keys()).copy()
        population_idxes = np.array(population_idxes)
        traffic_sent_per_population = np.zeros(len(population_idxes))
        for dst in range(self.N):
            for p_rel_idx in range(len(population_idxes)):
                traffic_sent_per_population[p_rel_idx] += self.traffic_base_population[dst][src_gpu_idx][p_rel_idx]

        # 排序并选出流量小且未被拆分的population
        sort_idx = np.argsort(traffic_sent_per_population)
        # idx = np.random.randint(0, len(sort_idx))
        idx = 0
        p_abs_idx = population_idxes[sort_idx[idx]]
        if self.map_table[src_gpu_idx][p_abs_idx] != 1:
            idx = (idx + 1) % len(sort_idx)
            p_abs_idx = population_idxes[sort_idx[idx]]

        print("small out traffic cortical:", src_gpu_idx, p_abs_idx)
        print("smallest traffic:", traffic_sent_per_population[np.where(population_idxes == p_abs_idx)[0][0]])

        return {'src': src_gpu_idx, 'dst': x, 'p_idx': p_abs_idx}

    def find_cortical_with_large_traffic_in(self):
        # old method(before 2021.12.07)
        x = self.random_selection_max(self.traffic_in_per_gpu, 1)
        y = x

        sizes_of_cortical = list()
        for cortical_idx in self.map_table[x]:
            sizes_of_cortical.append(self.size[cortical_idx])

        z = self.random_selection_max(sizes_of_cortical, 2)
        # print("large in traffic cortical:", x, y, z)

        print("large in traffic cortical:", x, y, z)

        return {'src': y, 'dst': x, 'p_idx': z}

    def find_cortical_with_small_traffic_in(self, max_i=3, max_j=0, max_k=3):
        # old method(before 2021.12.07)
        x = self.random_selection_min(self.traffic_in_per_gpu, max_i)
        y = x

        sizes_of_cortical = list()
        for cortical_idx in self.map_table[x]:
            sizes_of_cortical.append(self.size[cortical_idx])

        z = self.random_selection_min(sizes_of_cortical, max_k)
        print("small in traffic cortical:", x, y, z)

        return {'src': y, 'dst': x, 'p_idx': z}

    def find_cortical_with_large_size_degree(self):
        pass

    def find_cortical_with_small_size_degree(self):
        pass

    def update_map_table_out(self, p1, p2):
        abs_p_idx1 = p1['p_idx']
        abs_p_idx2 = p2['p_idx']

        self.map_table[p1['src']][abs_p_idx2] = self.map_table[p1['src']].pop(abs_p_idx1)
        self.map_table[p2['src']][abs_p_idx1] = self.map_table[p2['src']].pop(abs_p_idx2)

    def update_map_table_in(self, cortical_1, cortical_2):
        abs_cortical_idx1 = self.map_table[cortical_1['dst']][cortical_1['p_idx']]
        abs_cortical_idx2 = self.map_table[cortical_2['dst']][cortical_2['p_idx']]

        self.map_table[cortical_1['dst']].pop(cortical_1['p_idx'])
        self.map_table[cortical_1['dst']].append(abs_cortical_idx2)

        self.map_table[cortical_2['dst']].pop(cortical_2['p_idx'])
        self.map_table[cortical_2['dst']].append(abs_cortical_idx1)

    def recv_and_update_column(self, col):
        for row_idx in range(self.N):
            msg = self.comm.recv(source=row_idx % (self.comm_size - 1), tag=0)
            self.traffic_base_gpu[row_idx][col] = sum(msg)
            self.traffic_base_population[row_idx][col] = msg

    def recv_and_update_row(self, row):
        for col_idx in range(self.N):
            msg = self.comm.recv(source=col_idx % (self.comm_size - 1), tag=1)
            self.traffic_base_gpu[row][col_idx] = sum(msg)
            self.traffic_base_population[row][col_idx] = msg

    def update_traffic_master(self, cortical_1, cortical_2):
        # print('Begin to update...')
        self.recv_and_update_column(cortical_1['src'])
        # print('cortical1 col received')
        self.recv_and_update_column(cortical_2['src'])
        # print('cortical2 col received')
        self.recv_and_update_row(cortical_1['src'])
        # print('cortical1 row received')
        self.recv_and_update_row(cortical_2['src'])
        # print('cortical2 row received')


class GenerateMapParallelSlave(Parallelism, GenerateMap):
    def __init__(self):
        super().__init__()

        dcu_name = "cuda:" + str(self.rank % 4)
        self.device = torch.device(dcu_name)
        self.map_table = None

    def sample(self, sample_range, sample_times, n_slice):
        data = list()
        for i in range(n_slice):
            random_sample = torch.randint(0, int(sample_range), (int(sample_times / n_slice),), device=self.device)
            temp = torch.unique(random_sample.clone())
            data.append(temp)
            del temp
            torch.cuda.empty_cache()
        # print("%.fMB VRAM allocated." % (torch.cuda.memory_allocated(device=device) / 1000000))

        new_data = torch.cat(data)
        new_data = torch.unique(new_data)
        traffic = torch.unique(new_data).numel()
        return traffic

    def compute_traffic_between_two_gpu(self, gpu_out_idx, gpu_in_idx):
        traffic_gpu_to_gpu = 0
        traffic_src_to_gpu_lst = list()
        for population_out_idx in self.map_table[gpu_out_idx].keys():
            traffic_src_to_dst = list()
            for population_in_idx in self.map_table[gpu_in_idx].keys():
                if self.conn_version[0:5] == 'voxel':
                    conn_number_estimate = self.neuron_number * self.size[population_in_idx] * \
                                           self.map_table[gpu_in_idx][population_in_idx] * self.degree[
                                               population_in_idx] * self.conn[population_in_idx][population_out_idx]
                    traffic_src_to_dst.append(conn_number_estimate)
                else:  # cortical version of conn
                    key = int(population_in_idx * self.n + population_out_idx)
                    if key in self.conn_dict:
                        conn_number_estimate = self.neuron_number * self.size[population_in_idx] * \
                                               self.map_table[gpu_in_idx][population_in_idx] * self.degree[
                                                   population_in_idx] * self.conn_dict[key]
                        traffic_src_to_dst.append(conn_number_estimate)
                    else:
                        traffic_src_to_dst.append(0)

            sample_range = int(self.neuron_number * self.size[population_out_idx] * self.map_table[gpu_out_idx][
                population_out_idx])
            sample_times = int(np.sum(traffic_src_to_dst))

            torch.cuda.empty_cache()
            n = 1
            if sample_range > 2e7 or sample_times > 2e8:
                n = 10
                # print("range:", sample_range, " times:", sample_times)
            traffic_src_to_gpu = self.sample(sample_range, sample_times, n_slice=n)
            traffic_gpu_to_gpu += traffic_src_to_gpu
            traffic_src_to_gpu_lst.append(traffic_src_to_gpu)
        return traffic_src_to_gpu_lst

    def compute_and_send_col(self, col):
        # compute traffic_base_dcu by column
        row_idx_to_compute = self.allocate_idx_to_calculate()
        for row_idx in row_idx_to_compute:
            msg = self.compute_traffic_between_two_gpu(col, row_idx)
            self.comm.send(msg, dest=self.master_rank, tag=0)

    def compute_and_send_row(self, row):
        # compute traffic_base_dcu by column
        col_idx_to_compute = self.allocate_idx_to_calculate()
        for col_idx in col_idx_to_compute:
            msg = self.compute_traffic_between_two_gpu(col_idx, row)
            self.comm.send(msg, dest=self.master_rank, tag=1)

    def update_traffic_slave(self, cortical_1, cortical_2):
        self.compute_and_send_col(cortical_1['src'])
        self.compute_and_send_col(cortical_2['src'])
        self.compute_and_send_row(cortical_1['src'])
        self.compute_and_send_row(cortical_2['src'])


class GenerateMapParallel(GenerateMapParallelMaster, GenerateMapParallelSlave):
    def __init__(self):
        super().__init__()

        self.base_map_version = 'map_' + str(self.N) + '_v3_' + self.conn_version
        self.new_map_version = 'map_' + str(self.N) + '_v3_1_' + self.conn_version
        self.figure_save_path = self.root_path + "tables/traffic_table/2022_0315_noon2/"

        self.map_table = self.read_map_pkl('map_10000_1.05_cortical_v2_without_invalid_idx.pkl')
        # self.map_table = self.read_map_pkl(self.map_root + self.base_map_version + '.pkl')
        # self.map_table = self.map_table_normal_to_split(self.map_table)

        self.traffic_base_dcu_saving_path = self.traffic_table_root + 'traffic_table_base_dcu_' + self.new_map_version + '.npy'
        self.map_table_saving_path = self.map_root + self.new_map_version + '.pkl'
        self.map_table_without_invalid_idx_saving_path = self.map_root + self.new_map_version + '_without_invalid_idx.pkl'

        self.target = 3
        self.max_iter_cnt = int(2e2)

        self.show_traffic_interval = 1

        self.iter_cnt = 0
        self.best_map = {'map_table': list(), 'out': 1e4, 'in': 1e4}

        if self.rank == self.master_rank:
            self.show_basic_information()
            print()
            print("#" * 41)
            print("Base map version:", self.base_map_version)
            print("New map version: ", self.new_map_version)
            print("Target:", self.target)
            print("Max iteration:", self.max_iter_cnt)
            print("#" * 41)

            if not os.path.exists(self.figure_save_path):
                os.mkdir(self.figure_save_path)
            self.initialize_master()

        self.comm.barrier()

    @staticmethod
    def obj_function(map_table_dict):
        return map_table_dict['out'] + map_table_dict['in']
        # return map_table_dict['out']

    def iteration_stop_condition(self):
        top_k = 10

        # compute the current map's out traffic max / average
        max_traffic_out = np.max(self.traffic_out_per_gpu)
        average_traffic_out = np.average(self.traffic_out_per_gpu)
        object_out = max_traffic_out / average_traffic_out

        # compute the current map's in traffic max / average
        max_traffic_in = np.max(self.traffic_in_per_gpu)
        average_traffic_in = np.average(self.traffic_in_per_gpu)
        object_in = max_traffic_in / average_traffic_in

        if self.obj_function(self.best_map) > object_out + object_in:
            # if self.obj_function(self.best_map) > object_out:
            self.best_map = {'map_table': self.map_table, 'out': object_out, 'in': object_in}

        print("####### iter %d: " % self.iter_cnt)
        print("out: max / average = %.4f, best = %.4f" % (object_out, self.best_map['out']))
        print("in:  max / average = %.4f, best = %.4f" % (object_in, self.best_map['in']))
        print('top %d out: ' % top_k)
        top_k_traffic = self.traffic_out_per_gpu[
                            np.ix_(np.argsort(self.traffic_out_per_gpu)[-top_k:])] / average_traffic_out
        top_k_traffic = np.array([round(i, 2) for i in top_k_traffic])
        print(top_k_traffic)

        stop_condition = self.obj_function(self.best_map) > self.target and self.iter_cnt < self.max_iter_cnt

        return stop_condition

    def save_traffic_during_iteration(self):
        out_filename = self.figure_save_path + "traffic_out" + str(self.iter_cnt) + ".npy"
        in_file_name = self.figure_save_path + "traffic_in" + str(self.iter_cnt) + ".npy"

        np.save(out_filename, self.traffic_out_per_gpu)
        np.save(in_file_name, self.traffic_in_per_gpu)

    def update_in_traffic(self):
        p_1, p_2 = None, None
        if self.rank == self.master_rank:
            p_1 = self.find_cortical_with_large_traffic_in()
            p_2 = self.find_cortical_with_small_traffic_in()
        p_1 = self.comm.bcast(p_1, root=self.master_rank)
        p_2 = self.comm.bcast(p_2, root=self.master_rank)

        self.update_map_table_in(p_1, p_2)

        if self.rank == self.master_rank:
            self.update_traffic_master(p_1, p_2)
            self.cal_traffic()  # about 2 seconds
            if self.iter_cnt % self.show_traffic_interval == 0:
                self.save_traffic_during_iteration()
        else:
            self.update_traffic_slave(p_1, p_2)

        self.comm.barrier()

    def update_out_traffic(self):
        p_1, p_2 = None, None
        if self.rank == self.master_rank:
            p_1 = self.find_cortical_with_large_traffic_out(max_i=3, max_j=1)
            p_2 = self.find_cortical_with_small_traffic_out()
        p_1 = self.comm.bcast(p_1, root=self.master_rank)
        p_2 = self.comm.bcast(p_2, root=self.master_rank)

        self.update_map_table_out(p_1, p_2)

        if self.rank == self.master_rank:
            self.update_traffic_master(p_1, p_2)
            self.cal_traffic()  # about 2 seconds
            if self.iter_cnt % self.show_traffic_interval == 0:
                self.save_traffic_during_iteration()
        else:
            self.update_traffic_slave(p_1, p_2)

        self.comm.barrier()

    def update_size_degree(self):
        pass

    def iterate(self):
        self.update_out_traffic()
        # self.update_in_traffic()
        # self.update_size_degree()

    def generate_map_parallel(self):
        time_program_start = time.time()

        # loop_mark为循环是否继续的条件，由主进程计算并广播给所有子进程
        loop_mark = None
        if self.rank == self.master_rank:
            loop_mark = self.iteration_stop_condition()
        loop_mark = self.comm.bcast(loop_mark, root=self.master_rank)

        while loop_mark:
            time_iter_start = time.time()
            self.iter_cnt += 1

            self.iterate()  # 进行一次迭代

            if self.rank == self.master_rank:
                loop_mark = self.iteration_stop_condition()
            loop_mark = self.comm.bcast(loop_mark, root=self.master_rank)

            time_iter_end = time.time()
            if self.rank == self.master_rank:
                print("%.4f consumed." % (time_iter_end - time_iter_start))

        time_program_end = time.time()

        # save the result
        if self.rank == self.master_rank:
            print('Map generated. %.2fs consumed.' % (time_program_end - time_program_start))

            np.save(self.traffic_base_dcu_saving_path, self.traffic_base_gpu)
            print(self.traffic_base_dcu_saving_path + " saved. ")

            sum_row, sum_col = np.zeros(self.N), np.zeros(self.N)
            for i in range(self.N):
                inside_node_idx = np.arange(i // 4 * 4, i // 4 * 4 + 4)
                sum_col[i] = np.sum(self.traffic_base_gpu[:, i])
                sum_row[i] = np.sum(self.traffic_base_gpu[i, :])

                for idx in inside_node_idx:
                    sum_col[i] -= self.traffic_base_gpu[idx][i]
                    sum_row[i] -= self.traffic_base_gpu[i][idx]

            self.draw(sum_col, "src")
            self.draw(sum_row, "dst")
            # print('out: %.4f, in: %.4f' % (self.best_map['out'], self.best_map['in']))

            if self.map_table_without_invalid_idx_saving_path is not None:
                self.save_map_pkl(self.best_map['map_table'], self.map_table_without_invalid_idx_saving_path)

            self.map_table = self.map_table_transfer_split(self.N, self.map_table)

            if self.map_table_path is not None:
                self.save_map_pkl(self.map_table, self.map_table_saving_path)

    def draw(self, traffic_data, figure_name):
        x = np.arange(self.N)
        plt.figure(figsize=(10, 6), dpi=200)
        plt.title('max = %.2e, avg. = %.2e, max / avg. = %.2f' % (np.max(traffic_data), np.average(traffic_data),
                                                                  np.max(traffic_data) / np.average(traffic_data)),
                  fontsize=17)
        plt.xlabel('Process ID', fontsize=20)
        plt.ylabel('Spike Count', fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        # plt.scatter(x, msg_cnt,s=13)
        plt.plot(x, traffic_data)
        plt.plot(x, np.full(self.N, np.average(traffic_data)), linestyle='--', color='black', linewidth=2,
                 label='1x avg.')
        if figure_name == "src":
            plt.plot(x, np.full(self.N, np.average(traffic_data) * 5), linestyle='--', color='red',
                     linewidth=2, label='5x avg.')
        plt.legend(fontsize=17)
        # plt.plot(x, np.full(N, np.max(input)),linestyle='-',color='black',linewidth=2)
        plt.savefig('../tables/map_table/' + self.new_map_version + "_" + figure_name + '.png')
        print('../tables/map_table/' + self.new_map_version + "_" + figure_name + '.png saved.')

        print("" + self.new_map_version + "_" + figure_name + ":")
        print("Max: %.4e" % np.max(traffic_data))
        print("Average: %.4e" % np.average(traffic_data))
        print("Max / average: %.4e" % (np.max(traffic_data) / np.average(traffic_data)))
        print()




if __name__ == "__main__":
    Job = GenerateMap()
    Job.test()
    Job.generate_map_base_population_out_traffic(max_rate=1.5, max_iter_times=int(1e6), show_iter_process=True)
    # Job.generate_map_split(max_rate=1.05, max_iter_times=int(0.5e5), k_traffic=100, k_split=2,
    #                        show_iter_process=True)
    # Job.generate_my_map_new(max_rate=1.07, max_iter_times=int(1.0e6))
    # Job = GenerateMapParallel()
    # Job.generate_map_parallel()
