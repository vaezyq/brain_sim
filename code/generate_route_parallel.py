"""
Generate route table in parallel, CPU only.

When N = 16000, comm_size = 1601:
- It takes about 75 seconds for one iteration and 12.5 minutes for the whole process.

When N = 8000, comm_size = 1601:
- It takes about 17 seconds for one iteration
"""

import pickle
import numpy as np
import time
from generate_route import GenerateRoute
from mpi4py import MPI


def allocate_rows_to_generate(number_of_dcu, number_of_process, process_rank):
    """
    Allocate rows for processes.
    For example, suppose there are 100 dcus, and 11 processes in total, for the 11th process is responsible for
    gathering results, 10 processes participate in the computing process, which means 1 process is responsible
    for 10 rows' computing. Thus, the rows charged by process k is [k, 10+k, 20+k, ... 90+k]. That is,
    allocate_columns_to_calculate(100, 10, 2) = [2, 12, 22, ..., 92]
    """
    columns = list()
    for idx in range(int(number_of_dcu / number_of_process)):
        columns.append(idx * number_of_process + process_rank)

    return columns


if __name__ == '__main__':
    BioInfo = GenerateRoute()
    number_of_group = BioInfo.number_of_groups
    gpu_per_group = BioInfo.n_gpu_per_group
    assert number_of_group * gpu_per_group == BioInfo.N

    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    rank = comm.Get_rank()

    rows_per_process = int(BioInfo.N / (comm_size - 1))

    if rank == comm_size - 1:
        print('#########################################')
        print('MPI initialize Complete. comm_size = %d' % comm_size)
        # print('Rank %d is the master rank responsible for collecting and storing results.' % (comm_size-1))
        print('#########################################\n')
        BioInfo.show_basic_information()

    comm.barrier()

    '''
    Rank comm_size-1 have the following three functions:
    1. Receive the results calculated in other ranks;
    2. Merge the results received from other ranks into the final route table;
    3. Store the route table. 

    Ranks apart from rank comm_size-1 have the following functions:
    1. Generate route table **by rows**;
    2. Send the result to rank comm_size-1
    '''

    if rank == comm_size - 1:
        route_table = np.zeros((BioInfo.N, BioInfo.N), dtype=int)
        for i in range(rows_per_process):
            time1 = time.time()
            for j in range(comm_size - 1):
                msg = comm.recv(source=j)
                route_table[(comm_size - 1) * i + j, :] = msg
            time2 = time.time()
            print('i = %d/%d, %.4f consumed.' % (i + 1, rows_per_process, time2 - time1))
        np.save(BioInfo.route_path + 'route.npy', route_table)
        print(BioInfo.route_path + 'route.npy saved.')

        BioInfo.save_route_json(route_table, BioInfo.route_path + 'route_dense.json')
    else:
        rows_to_generate = allocate_rows_to_generate(BioInfo.N, comm_size - 1, rank)

        with open(BioInfo.route_path + 'forwarding_table.pickle', 'rb') as f:
            forwarding_table = pickle.load(f)

        # generate route table by row
        for dcu_out_idx in rows_to_generate:
            route_for_a_row = np.zeros((BioInfo.N,), dtype=int)
            row, col = dcu_out_idx % gpu_per_group, dcu_out_idx // gpu_per_group

            for dcu_in_idx in range(BioInfo.N):
                if dcu_out_idx % gpu_per_group == dcu_in_idx % gpu_per_group:  # 在同一个group
                    route_for_a_row[dcu_in_idx] = dcu_out_idx
                elif dcu_in_idx in forwarding_table[row][col]:  # 是本dcu负责转发的
                    route_for_a_row[dcu_in_idx] = dcu_out_idx
                else:  # 找到负责转发的
                    dst = -1
                    for i in range(number_of_group):
                        if dcu_in_idx in forwarding_table[row][i]:
                            dst = gpu_per_group * i + row
                    assert dst != -1
                    route_for_a_row[dcu_in_idx] = dst

            comm.send(route_for_a_row, dest=comm_size - 1)
            # print('row %d sent.' % dcu_out_idx)
