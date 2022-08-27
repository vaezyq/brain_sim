import os.path

import numpy as np
import matplotlib.pyplot as plt
import pickle
import time


class AdjustMapTable:
    def __init__(self):
        self.new_map_version = 'map_10000_v5_cortical_v2'
        self.old_map_version = 'map_10000_v4_cortical_v2'

        self.map_table_path_ = '../tables/map_table/' + self.old_map_version + '_without_invalid_idx.pkl'
        self.traffic_table_base_dcu_path_ = '../tables/traffic_table/traffic_table_base_dcu_' \
                                            + self.old_map_version + '.npy'
        self.traffic_base_cortical_path_ = '../tables/traffic_table/traffic_base_cortical_' \
                                           + self.old_map_version + '.pkl'
        self.origin_size_path_ = '../tables/conn_table/cortical_v2/origin_size.npy'

        self.new_map_saving_path_ = '../tables/map_table/' + self.new_map_version + '.pkl'
        self.new_map_without_invalid_idx_saving_path_ = '../tables/map_table/' \
                                                        + self.new_map_version + '_without_invalid_idx.pkl'

        self.N_ = None
        self.n_ = None
        self.map_table_ = None
        self.traffic_table_base_dcu_ = None
        self.traffic_base_cortical_ = None
        self.origin_size_ = None

        self.initialize()
        self.show_information()

    def initialize(self):
        with open(self.map_table_path_, 'rb') as f:
            self.map_table_ = pickle.load(f)
        self.traffic_table_base_dcu_ = np.load(self.traffic_table_base_dcu_path_)

        self.origin_size_ = np.load(self.origin_size_path_)

        self.N_ = len(self.map_table_)
        self.n_ = 0
        for i in range(self.N_):
            self.n_ += len(self.map_table_[str(i)])

    def show_information(self):
        print("Old map version:", self.old_map_version)
        print("New map version:", self.new_map_version)
        print("Number of GPUs:", self.N_)
        print("Number of population:", self.n_)

    def compute_traffic_per_voxel(self, max_larger_times=10):
        traffic_per_cortical = np.zeros(self.n_)
        file_name = '../tables/traffic_table/traffic_per_cortical_' + self.old_map_version + '.npy'
        if os.path.exists(file_name):
            traffic_per_cortical = np.load(file_name)
        else:
            with open(self.traffic_base_cortical_path_, 'rb') as f:
                self.traffic_base_cortical_ = pickle.load(f)

            time1 = time.time()
            for gpu_idx in range(self.N_):
                for dst_idx in range(self.N_):
                    if gpu_idx != dst_idx:
                        for i in range(len(self.map_table_[str(gpu_idx)])):
                            cortical_idx = self.map_table_[str(gpu_idx)][i]
                            traffic_per_cortical[cortical_idx] += self.traffic_base_cortical_[dst_idx][gpu_idx][i]
                if gpu_idx % 2000 == 0:
                    print(gpu_idx)
            time2 = time.time()
            print('%.2f' % (time2 - time1))
            np.save(file_name, traffic_per_cortical)
            print(file_name + ' saved.')

        average = np.sum(traffic_per_cortical) / self.N_
        cnt_large = np.zeros(max_larger_times + 1)

        for i in range(self.n_):
            for j in range(1, max_larger_times + 1):
                if traffic_per_cortical[i] > j * average:
                    cnt_large[j] += 1

        for i in range(1, max_larger_times + 1):
            print("%d: %d" % (i, cnt_large[i]))

        return traffic_per_cortical

    def shuffle_map_table(self):
        pass

    def update_map_table(self):
        num_of_seperated_population = 8
        traffic_per_voxel = self.compute_traffic_per_voxel()

        map_table_traffic = list()
        traffic_per_gpu = np.empty(self.N_)
        for i in range(self.N_):
            lst = list()
            for j in range(len(self.map_table_[str(i)])):
                lst.append(traffic_per_voxel[self.map_table_[str(i)][j]])
            map_table_traffic.append(lst)
            traffic_per_gpu[i] = sum(lst)

        # 找出流量大的voxel
        voxel_idx_with_large_traffic = np.argsort(traffic_per_voxel)[-num_of_seperated_population:]
        voxel_idx_with_large_traffic.sort()

        # 在map_table中删除这些voxel
        for i in range(num_of_seperated_population):
            voxel_idx = voxel_idx_with_large_traffic[i]
            for j in range(len(self.map_table_)):
                if voxel_idx in self.map_table_[str(j)]:
                    self.map_table_[str(j)].remove(voxel_idx)
                    traffic_per_gpu[j] -= traffic_per_voxel[voxel_idx]
                    break

        print(np.sum(traffic_per_voxel) - np.sum(traffic_per_gpu))

        gpu_idx_with_min_traffic = np.argsort(traffic_per_gpu)[0:num_of_seperated_population]

        # 把前8个gpu所模拟的population分到流量小的gpu中去(不包括0-7号gpu)
        for i in range(num_of_seperated_population):
            gpu_idx = gpu_idx_with_min_traffic[i]
            for voxel_idx in self.map_table_[str(gpu_idx)]:
                temp = np.argsort(traffic_per_gpu)

                loc = num_of_seperated_population
                while temp[loc] in gpu_idx_with_min_traffic:
                    loc += 1

                # self.map_table_[str(i)].remove(voxel_idx)
                # traffic_per_gpu[i] -= traffic_per_voxel[voxel_idx]

                self.map_table_[str(temp[loc])].append(voxel_idx)
                traffic_per_gpu[temp[loc]] += traffic_per_voxel[voxel_idx]

        for i in range(num_of_seperated_population):
            gpu_idx = gpu_idx_with_min_traffic[i]
            voxel_idx = voxel_idx_with_large_traffic[i]
            self.map_table_[str(gpu_idx)] = list([voxel_idx])
            traffic_per_gpu[gpu_idx] = traffic_per_voxel[voxel_idx]

        print(np.sum(traffic_per_voxel) - np.sum(traffic_per_gpu))
        plt.figure(figsize=(10, 6), dpi=200)
        plt.title('src max / average: %.4f' % (np.max(traffic_per_gpu) / np.average(traffic_per_gpu)))
        plt.plot(traffic_per_gpu)
        plt.plot(np.arange(0, 10000), np.full(10000, np.average(traffic_per_gpu)), label='average')
        # plt.plot(np.arange(0, 10000), 3 * np.full(10000, np.average(traffic_per_gpu)), label='3x average')
        plt.plot(np.arange(0, 10000), 4 * np.full(10000, np.average(traffic_per_gpu)), label='4x average')
        plt.plot(np.arange(0, 10000), 5 * np.full(10000, np.average(traffic_per_gpu)), label='5x average')
        plt.legend(fontsize=14)
        plt.show()

        print()
        self.check_map_without_invalid_idx(self.map_table_)

        with open(self.new_map_without_invalid_idx_saving_path_, 'wb') as f:
            pickle.dump(self.map_table_, f)
        print(self.new_map_without_invalid_idx_saving_path_ + ' saved.')

        self.map_table_transfer()
        self.check_map(self.map_table_)
        with open(self.new_map_saving_path_, 'wb') as f:
            pickle.dump(self.map_table_, f)
        print(self.new_map_saving_path_ + ' saved.')

    def map_table_transfer(self):
        map_171508_to_227030 = list()
        for i in range(len(self.origin_size_)):
            if self.origin_size_[i] != 0:
                map_171508_to_227030.append(i)

        for i in range(len(self.map_table_)):
            for j in range(len(self.map_table_[str(i)])):
                self.map_table_[str(i)][j] = map_171508_to_227030[self.map_table_[str(i)][j]]

    def check_map_without_invalid_idx(self, map_table):
        voxel_sequential_idx = list()
        for i in range(self.N_):
            for voxel_idx in map_table[str(i)]:
                voxel_sequential_idx.append(voxel_idx)

        voxel_sequential_idx.sort()
        for i in range(self.n_):
            if voxel_sequential_idx[i] != i:
                print('ERROR!')
                break

    def check_map(self, map_table):
        voxel_sequential_idx = list()
        for i in range(self.N_):
            for voxel_idx in map_table[str(i)]:
                voxel_sequential_idx.append(voxel_idx)

        voxel_sequential_idx.sort()

        print(len(voxel_sequential_idx))
        print(voxel_sequential_idx[0:8])
        print(voxel_sequential_idx[8:16])
        print(voxel_sequential_idx[16:24])


if __name__ == '__main__':
    Job = AdjustMapTable()
    Job.update_map_table()
