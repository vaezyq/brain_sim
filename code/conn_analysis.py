import numpy as np
import pickle
import time

from parallelism import Parallelism
from data import Data


class ConnAnalysisParallel(Parallelism, Data):
    def __init__(self):
        super().__init__()

        self.out_traffic_per_population_path = self.conn_root + 'out_traffic_per_population.npy'

        if self.rank == self.master_rank:
            self.show_basic_information()
        self.comm.barrier()

    def compute_p2p_traffic_between_population(self, src_idx, dst_idx):
        hash_key = src_idx * self.n + dst_idx
        # p2p_traffic = self.neuron_number * self.size[dst_idx] * self.degree[dst_idx] * self.conn_dict[hash_key]
        p2p_traffic = self.neuron_number * self.size[dst_idx] * self.degree[dst_idx] * self.conn[dst_idx][src_idx]

        return p2p_traffic

    def find_population_with_max_traffic_parallel(self):
        if self.rank == self.master_rank:
            out_traffic_per_population = np.zeros(self.n)

            for population_idx in range(self.n):
                msg = self.comm.recv(source=population_idx % (self.comm_size - 1))
                out_traffic_per_population[population_idx] = msg

            np.save(self.out_traffic_per_population_path, out_traffic_per_population)
        else:
            population_idx_to_compute = self.allocate_population_idx_to_calculate(self.n)
            # 计算单个population发送到其他所有population的流量和（不并包）
            for src in population_idx_to_compute:
                time1 = time.time()

                p2p_traffics = np.zeros(self.n)
                for dst in range(self.n):
                    p2p_traffics[dst] = self.compute_p2p_traffic_between_population(src, dst)

                out_traffic_sum = np.sum(p2p_traffics)
                time2 = time.time()
                self.comm.send(out_traffic_sum, dest=self.master_rank)
                print("Population %d sent, %.f seconds consumed." % (src, time2 - time1))


class ConnAnalysis(Data):
    def __init__(self):
        super().__init__()

        self.out_traffic_per_population_path = self.conn_root + 'out_traffic_per_population.npy'
        self.out_traffic_per_population = np.load(self.out_traffic_per_population_path)

    def pick_population_idx_with_large_out_traffic(self, times):
        population_idxes = list()
        average = np.average(self.out_traffic_per_population)

        for idx in range(self.n):
            if self.out_traffic_per_population[idx] > times * average:
                population_idxes.append(idx)

        print("%d population's traffic is larger than %dx average." % (len(population_idxes), times))

        return population_idxes


if __name__ == "__main__":
    Job = ConnAnalysisParallel()
    Job.find_population_with_max_traffic_parallel()
