"""
This file calculates the traffic between any two dcus without any route strategy based on the map table.
N = 16000, comm_size = 1600:
- 500s needed for one iteration and 5000s for ten iterations.
- 5400 seconds(1.5 hour) needed to run the whole program.
"""

import numpy as np
import time
from map_analysis import MapAnalysis
from mpi4py import MPI
import torch


def allocate_columns_to_calculate(number_of_dcu, number_of_process, process_rank):
    """
    Allocate columns for processes.
    For example, suppose there are 100 dcus, and 11 processes in total, for the 11th process is responsible for
    gathering results, 10 processes participate in the computing process, which means 1 process is responsible
    for 10 columns' computing. Thus, the columns charged by process k is [k, 10+k, 20+k, ... 90+k]. That is,
    allocate_columns_to_calculate(100, 10, 2) = [2, 12, 22, ..., 92]
    """
    columns = list()
    for idx in range(int(number_of_dcu / number_of_process)):
        columns.append(idx * number_of_process + process_rank)

    return columns


def calculate_traffic_between_two_dcus(map_info, map_table, device, out_idx, in_idx):
    output = list()

    traffic_dcu_to_dcu = 0
    for cortical_out_idx in map_table[out_idx]:
        traffic_src_to_dst = list()
        for cortical_in_idx in map_table[in_idx]:
            key = int(cortical_in_idx * map_info.n + cortical_out_idx)
            conn_number_estimate = map_info.neuron_number * map_info.size[cortical_in_idx] * \
                                   map_info.degree[cortical_in_idx] * map_info.conn[cortical_in_idx][cortical_out_idx]
            traffic_src_to_dst.append(conn_number_estimate)

        sample_range = int(map_info.neuron_number * map_info.size[cortical_out_idx])
        sample_times = int(np.sum(traffic_src_to_dst))
        # print()
        # print("Sample range: %d, Sample times: %d" % (sample_range, sample_times))

        traffic_src_to_dcu = torch.unique(
            torch.randint(0, sample_range, (sample_times,), device=device).clone()).numel()
        torch.cuda.empty_cache()
        traffic_dcu_to_dcu += traffic_src_to_dcu
        output.append(traffic_src_to_dcu)

    return traffic_dcu_to_dcu, output


def check_traffic(map_info):
    sum_row, sum_col = np.zeros(map_info.N), np.zeros(map_info.N)
    for i in range(map_info.N):
        sum_row[i] = np.sum(traffic_table_base_dcu[i, :]) - traffic_table_base_dcu[i][i]
        sum_col[i] = np.sum(traffic_table_base_dcu[:, i]) - traffic_table_base_dcu[i][i]

    print('in: ', np.max(sum_row) / np.average(sum_row))
    print('out: ', np.max(sum_col) / np.average(sum_col))


if __name__ == '__main__':
    Map_info = MapAnalysis()
    map_table = Map_info.map_table_without_invalid_idx

    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    rank = comm.Get_rank()

    columns_per_process = int(Map_info.N / (comm_size - 1))

    if rank == comm_size - 1:
        print('#########################################')
        print('MPI initialize Complete. comm_size = %d' % comm_size)
        # print('Rank %d is the master rank responsible for collecting and storing results.' % (comm_size-1))

        print("First 10 rows of map_table: ")
        for i in range(10):
            print(map_table[i])
        print('#########################################\n')
        Map_info.show_basic_information()

    comm.barrier()

    '''
    Rank comm_size-1 have the following three functions:
    1. Receive the results calculated in other ranks;
    2. Merge the results received from other ranks into the final traffic matrix based dcu;
    3. Store the final matrix. 

    Ranks apart from rank comm_size-1 have the following functions:
    1. Calculate the traffic from source dcu to destination dcus **by column**;
    2. Send the result to rank comm_size-1
    '''

    if rank == comm_size - 1:
        time_start = time.time()
        traffic_table_base_dcu = np.zeros((Map_info.N, Map_info.N))
        traffic_base_cortical = [[None] * Map_info.N for _ in range(Map_info.N)]

        for i in range(columns_per_process):
            time1 = time.time()
            for j in range(comm_size - 1):
                col = (comm_size - 1) * i + j

                msg = comm.recv(source=j, tag=0)
                traffic_table_base_dcu[:, col] = msg

                msg = comm.recv(source=j, tag=1)
                for k in range(Map_info.N):
                    traffic_base_cortical[k][col] = msg[k]

            time2 = time.time()
            print('i = %d/%d, %.4f consumed.' % (i + 1, columns_per_process, time2 - time1))

        time.sleep(10)

        # print(traffic_table_base_dcu)
        # print(traffic_table_base_dcu.shape)

        npy_name = 'traffic_table_base_dcu_' + Map_info.map_version + '.npy'
        np.save(Map_info.traffic_table_root + npy_name, traffic_table_base_dcu)
        print(Map_info.traffic_table_root + npy_name + " saved.")
        print(np.sum(traffic_table_base_dcu))

        pkl_name = 'traffic_base_cortical_' + Map_info.map_version + '.pkl'
        import pickle
        with open(Map_info.traffic_table_root + pkl_name, 'wb') as f:
            pickle.dump(traffic_base_cortical, f)
        print(Map_info.traffic_table_root + pkl_name + " saved.")

        time_end = time.time()
        print("%d nodes used. " % ((comm_size - 1) / 32 + 1))
        check_traffic(Map_info)
        print("%.f seconds consumed." % (time_end - time_start))
    else:
        columns_to_calculate = allocate_columns_to_calculate(Map_info.N, comm_size - 1, rank)
        for dcu_out_idx in columns_to_calculate:
            # calculate the traffic, stored as a (N, 1) array
            traffic_base_dcu_for_a_column = np.zeros((Map_info.N,))
            traffic_base_cortical_for_a_column = list()

            dcu_name = "cuda:" + str(rank % 4)
            torch_device = torch.device(dcu_name if torch.cuda.is_available() else "cpu")

            # calculate traffic from one dcu to one dcu
            time1 = time.time()
            for dcu_in_idx in range(Map_info.N):
                traffic_base_dcu_for_a_column[dcu_in_idx], output = \
                    calculate_traffic_between_two_dcus(Map_info, map_table, torch_device, dcu_out_idx, dcu_in_idx)
                traffic_base_cortical_for_a_column.append(output)

            time2 = time.time()
            comm.send(traffic_base_dcu_for_a_column, dest=comm_size - 1, tag=0)
            comm.send(traffic_base_cortical_for_a_column, dest=comm_size-1, tag=1)
            # print('Col %d: %.4fs consumed.' % (dcu_out_idx, time2 - time1))
