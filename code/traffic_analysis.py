import numpy as np
import pickle
import time

from parallelism import Parallelism
from data import Data


class TrafficAnalysis(Parallelism, Data):
    def __init__(self):
        super().__init__()

        traffic_root = self.root_path + "/tables/traffic_table/"
        file_name = "traffic_base_cortical_" + self.map_version + ".pkl"
        traffic_base_cortical_path = traffic_root + file_name

        time1 = time.time()
        with open(traffic_base_cortical_path, 'rb') as f:
            self.traffic_base_cortical_ = pickle.load(f)
        print(traffic_base_cortical_path + ' loaded.')
        time2 = time.time()
        if self.rank == self.master_rank:
            print("%.2f seconds consumed." % (time2 - time1))

        map_root = self.root_path + 'tables/map_table/'
        map_path = map_root + self.map_version + '.pkl'
        with open(map_path, 'rb') as f:
            self.map_table_ = pickle.load(f)
        if self.rank == self.master_rank:
            print("map table loaded.")

    def compute_traffic_per_population(self):
        traffic_per_population = np.zeros(self.n)

        if self.rank == self.master_rank:
            pass
        else:
            columns_to_compute = self.allocate_idx_to_calculate()
            for gpu_idx in columns_to_compute:
                for dst_idx in range(self.N):
                    if gpu_idx != dst_idx:
                        for i in range(len(self.map_table_[str(gpu_idx)])):
                            population_idx = self.map_table_[str(gpu_idx)][i]
                            traffic_per_population[population_idx] += self.traffic_base_cortical_[dst_idx][gpu_idx][i]

