from mpi4py import MPI
import numpy as np
import torch
import time
from parallelism import Parallelism
from generate_map import GenerateMap


class GenerateMapParallelSlave(Parallelism, GenerateMap):
    def __init__(self):
        super().__init__()

        dcu_name = "cuda:" + str(self.rank % 4)
        self.device = torch.device(dcu_name if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.map_table = None

    def sample(self, sample_range, sample_times, n_slice=10):
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

    def compute_traffic_between_two_dcus(self, src_idx, dst_idx):
        traffic_dcu_to_dcu = list()
        for cortical_src_idx in self.map_table[src_idx]:
            # print('begin cortical %d' % cortical_src_idx)
            traffic_src_to_dst = list()
            for cortical_dst_idx in self.map_table[dst_idx]:
                conn_number_estimate = self.neuron_number * self.size[cortical_dst_idx] * \
                                       self.degree[cortical_dst_idx] * self.conn[cortical_dst_idx][
                                           cortical_src_idx]
                traffic_src_to_dst.append(conn_number_estimate)

            sample_range = int(self.neuron_number * self.size[cortical_src_idx])
            sample_times = int(np.sum(traffic_src_to_dst))

            torch.cuda.empty_cache()
            n = 1
            if sample_range > 1e7 or sample_times > 1e8:
                n = 10
                # print("range:", sample_range, " times:", sample_times)
            traffic_dcu_to_dcu.append(self.sample(sample_range, sample_times, n_slice=n))

        return traffic_dcu_to_dcu

    def compute_and_send_col(self, col):
        # compute traffic_base_dcu by column
        row_idx_to_compute = self.allocate_idx_to_calculate()
        for row_idx in row_idx_to_compute:
            msg = self.compute_traffic_between_two_dcus(col, row_idx)
            self.comm.send(msg, dest=self.master_rank, tag=0)

    def compute_and_send_row(self, row):
        # compute traffic_base_dcu by column
        col_idx_to_compute = self.allocate_idx_to_calculate()
        for col_idx in col_idx_to_compute:
            msg = self.compute_traffic_between_two_dcus(col_idx, row)
            self.comm.send(msg, dest=self.master_rank, tag=1)

    def update_traffic_slave(self, cortical_1, cortical_2):
        self.compute_and_send_col(cortical_1['src'])
        self.compute_and_send_col(cortical_2['src'])
        self.compute_and_send_row(cortical_1['src'])
        self.compute_and_send_row(cortical_2['src'])
