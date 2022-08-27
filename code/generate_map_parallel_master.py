import os
import numpy as np
import time
import pickle
from parallelism import Parallelism
from generate_map import GenerateMap


class GenerateMapParallelMaster(Parallelism, GenerateMap):
    def __init__(self):
        super().__init__()

        self.figure_save_path = self.root_path + "tables/traffic_table/1211_noon4/"
        self.traffic_base_dcu_path = self.traffic_table_root + "traffic_table_base_dcu_" + self.map_version + ".npy"
        self.traffic_base_cortical_path = self.traffic_table_root + "traffic_base_cortical_" + self.map_version + ".pkl"

        self.map_table = None
        self.traffic_base_dcu = None
        self.traffic_base_cortical = None
        self.origin_out, self.origin_in = None, None
        self.traffic_out_per_dcu, self.traffic_in_per_dcu = None, None

        if self.rank == self.master_rank:
            if not os.path.exists(self.figure_save_path):
                os.mkdir(self.figure_save_path)
            self.initialize_master()
            self.show_basic_information()
            print("Base map version:", self.map_version)

    def initialize_master(self):
        time1 = time.time()
        self.traffic_base_dcu = np.load(self.traffic_base_dcu_path)
        print(self.traffic_base_dcu_path + " loaded.")

        with open(self.traffic_base_cortical_path, 'rb') as f:
            self.traffic_base_cortical = pickle.load(f)
        print(self.traffic_base_cortical_path + ' loaded.')

        self.traffic_out_per_dcu, self.traffic_in_per_dcu = np.empty(self.N), np.empty(self.N)
        self.cal_traffic()
        self.origin_out, self.origin_in = self.traffic_out_per_dcu, self.traffic_in_per_dcu
        np.save(self.figure_save_path + "origin_out.npy", self.origin_out)
        np.save(self.figure_save_path + "origin_in.npy", self.origin_in)

        time2 = time.time()
        print('Iteration initialization complete. %.2fs consumed' % (time2 - time1))

    # could be further accelerated
    def cal_traffic(self):
        """
        calculate traffic per dcu in and out
        :return:
        """
        for i in range(self.N):
            self.traffic_out_per_dcu[i] = np.sum(self.traffic_base_dcu[:, i]) - self.traffic_base_dcu[i][i]
            self.traffic_in_per_dcu[i] = np.sum(self.traffic_base_dcu[i, :]) - self.traffic_base_dcu[i][i]




    @staticmethod
    def random_selection_max(elements_to_choose, k):
        """
        :param elements_to_choose: Elements to be selected
        :param k: Pick one of the first k elements randomly
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

    def find_cortical_with_large_traffic_out(self, max_i=4, max_j=2, max_k=1):
        """
        max_i, max_j is better to be 1. If they are more than 1, it will cause copying and sorting of a array
        of size self.N, thus leading to a considerable consumption of time.
        """

        # old method(before 2021.12.07)
        # y = self.random_selection_max(self.traffic_out_per_dcu, max_i)
        # print("largest traffic:", y, self.traffic_out_per_dcu[y])
        # x = y
        # while x == y:
        #     x = self.random_selection_max(self.traffic_base_dcu[:, y], max_j)
        # z = self.random_selection_max(self.traffic_base_cortical[x][y], max_k)
        # print("large out traffic cortical:", x, y, z, self.size[self.map_table[y][z]])

        # new method
        y = self.random_selection_max(self.traffic_out_per_dcu, max_i)
        x = -1
        num_of_cortical = len(self.map_table[y])
        traffic_sent_per_cortical = np.zeros(num_of_cortical)
        for dst in range(self.N):
            for cortical_idx in range(num_of_cortical):
                traffic_sent_per_cortical[cortical_idx] += self.traffic_base_cortical[dst][y][cortical_idx]
        z = self.random_selection_max(traffic_sent_per_cortical, max_k)
        print("large out traffic cortical:", x, y, z)
        print("largest traffic:", np.max(traffic_sent_per_cortical))

        return {'src': y, 'dst': x, 'p_idx': z}

    def find_cortical_with_small_traffic_out(self, max_i=1, max_j=1, max_k=1):
        # old method(before 2021.12.07)
        # y = self.random_selection_min(self.traffic_out_per_dcu, max_i)
        # x = self.random_selection_min(self.traffic_base_dcu[:, y], max_j)
        # z = self.random_selection_min(self.traffic_base_cortical[x][y], max_k)
        # print("small out traffic cortical:", x, y, z)

        # new method
        y = self.random_selection_min(self.traffic_out_per_dcu, max_i)
        x = -1
        num_of_cortical = len(self.map_table[y])
        traffic_sent_per_cortical = np.zeros(num_of_cortical)
        for dst in range(self.N):
            for cortical_idx in range(num_of_cortical):
                traffic_sent_per_cortical[cortical_idx] += self.traffic_base_cortical[dst][y][cortical_idx]
        z = self.random_selection_min(traffic_sent_per_cortical, max_k)
        print("small out traffic cortical:", x, y, z)

        return {'src': y, 'dst': x, 'p_idx': z}

    def find_cortical_with_large_traffic_in(self):
        # old method(before 2021.12.07)
        x = self.random_selection_max(self.traffic_in_per_dcu, 1)
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
        x = self.random_selection_min(self.traffic_in_per_dcu, max_i)
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

    def update_map_table_out(self, cortical_1, cortical_2):
        abs_cortical_idx1 = self.map_table[cortical_1['src']][cortical_1['p_idx']]
        abs_cortical_idx2 = self.map_table[cortical_2['src']][cortical_2['p_idx']]

        self.map_table[cortical_1['src']].pop(cortical_1['p_idx'])
        self.map_table[cortical_1['src']].append(abs_cortical_idx2)

        self.map_table[cortical_2['src']].pop(cortical_2['p_idx'])
        self.map_table[cortical_2['src']].append(abs_cortical_idx1)

    def update_map_table_in(self, cortical_1, cortical_2):
        abs_cortical_idx1 = self.map_table[cortical_1['dst']][cortical_1['p_idx']]
        abs_cortical_idx2 = self.map_table[cortical_2['dst']][cortical_2['p_idx']]

        self.map_table[cortical_1['dst']].pop(cortical_1['p_idx'])
        self.map_table[cortical_1['dst']].append(abs_cortical_idx2)

        self.map_table[cortical_2['dst']].pop(cortical_2['p_idx'])
        self.map_table[cortical_2['dst']].append(abs_cortical_idx1)

    def recv_and_update_column(self, col):
        for row_idx in range(self.N):
            msg = self.comm.recv(source=row_idx % (self.comm_size-1), tag=0)
            self.traffic_base_dcu[row_idx][col] = sum(msg)
            self.traffic_base_cortical[row_idx][col] = msg

    def recv_and_update_row(self, row):
        for col_idx in range(self.N):
            msg = self.comm.recv(source=col_idx % (self.comm_size-1), tag=1)
            self.traffic_base_dcu[row][col_idx] = sum(msg)
            self.traffic_base_cortical[row][col_idx] = msg

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
