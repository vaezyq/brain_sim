import pickle
import numpy as np
import time
from route_analysis import RouteAnalysis
from mpi4py import MPI
import matplotlib.pyplot as plt


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


def show_traffic(level1_traffic, level2_traffic, name):
    print('level1:')
    print(np.min(level1_traffic), np.max(level1_traffic), np.average(level1_traffic))
    print('level2:')
    print(np.min(level2_traffic), np.max(level2_traffic), np.average(level2_traffic))

    plt.figure(figsize=(12, 6), dpi=200)
    plt.title('level1 traffic: max = %.f, average = %.f, max/average = %.2f' %
              (np.max(level1_traffic), np.average(level1_traffic),
               np.max(level1_traffic) / np.average(level1_traffic)))
    # plt.ylim(500000, 260000000)
    plt.plot(level1_traffic, linewidth=0.15)
    plt.savefig(BioInfo.route_path + 'parallel' + name + '_level1_traffic.png')
    # plt.show()

    plt.figure(figsize=(12, 6), dpi=200)
    plt.title('level2 traffic: max = %.f, average = %.f, max/average = %.2f' %
              (np.max(level2_traffic), np.average(level2_traffic),
               np.max(level2_traffic) / np.average(level2_traffic)))
    # plt.ylim(500000, 260000000)
    plt.plot(level2_traffic, linewidth=0.15)
    plt.savefig(BioInfo.route_path + 'parallel' + name + '_level2_traffic.png')
    # plt.show()


if __name__ == '__main__':
    BioInfo = RouteAnalysis()

    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    rank = comm.Get_rank()

    rows_per_process = int(BioInfo.N / (comm_size-1))

    if rank == comm_size-1:
        print('#########################################')
        print('MPI initialize Complete. comm_size = %d' % comm_size)
        # print('Rank %d is the master rank responsible for collecting and storing results.' % (comm_size-1))
        print('#########################################\n')

    comm.barrier()

    '''
    Rank comm_size-1 have the following three functions:
    1. Receive the results calculated in other ranks;
    2. Merge the results received from other ranks;
    3. Store the2-level traffic. 

    Ranks apart from rank comm_size-1 have the following functions:
    1. Calculate traffic **by rows**;
    2. Send the result to rank comm_size-1
    '''

    if rank == comm_size-1:
        time_start = time.time()
        level1_traffic_in, level2_traffic_in = np.zeros(BioInfo.N), np.zeros(BioInfo.N)
        level1_traffic_out, level2_traffic_out = np.zeros(BioInfo.N), np.zeros(BioInfo.N)

        for i in range(rows_per_process):
            time1 = time.time()
            for j in range(comm_size-1):
                level1_msg_out = comm.recv(source=j, tag=1)
                level1_traffic_out[(comm_size-1)*i + j] = level1_msg_out

                level2_msg_out = comm.recv(source=j, tag=2)
                for key in level2_msg_out:
                    level2_traffic_out[key] += level2_msg_out[key]

                level1_msg_in = comm.recv(source=j, tag=1)
                level1_traffic_in += level1_msg_in
                level2_msg_in = comm.recv(source=j, tag=2)
                level2_traffic_in += level2_msg_in
            time2 = time.time()
            print('i = %d/%d, %.4f consumed.' % (i, rows_per_process - 1, time2 - time1))

        show_traffic(level1_traffic_in, level2_traffic_in, name='in')
        show_traffic(level1_traffic_out, level2_traffic_out, name='out')
        time_end = time.time()
        print('Calculation complete. %.2f seconds consumed. ' % (time_end - time_start))
    else:
        rows_to_generate = allocate_rows_to_generate(BioInfo.N, comm_size-1, rank)

        route_table = BioInfo.route_table
        traffic_table_base_dcu = np.load('../tables/map_table/traffic_table_base_dcu_17280.npy')

        for row in rows_to_generate:
            # 先统计各个节点负责的dcu有哪些
            masters_and_slaves = dict()
            for j in range(BioInfo.N):
                key = route_table[row][j]
                if key not in masters_and_slaves:
                    masters_and_slaves[key] = list()
                masters_and_slaves[key].append(j)

            level1_traffic_in, level2_traffic_in = np.zeros(BioInfo.N), np.zeros(BioInfo.N)
            level1_traffic_out, level2_traffic_out = 0, dict()

            for j in range(BioInfo.N):
                key = route_table[row][j]

                level1_traffic_out += traffic_table_base_dcu[j][row]
                level1_traffic_in[j] = traffic_table_base_dcu[j][row]

                if j not in masters_and_slaves[row]:
                    if key not in level2_traffic_out:
                        level2_traffic_out[key] = 0

                    level2_traffic_out[key] += traffic_table_base_dcu[j][row]
                    level2_traffic_in[j] = traffic_table_base_dcu[j][row]

            comm.send(level1_traffic_out, dest=comm_size-1, tag=1)
            comm.send(level2_traffic_out, dest=comm_size-1, tag=2)
            comm.send(level1_traffic_in, dest=comm_size-1, tag=1)
            comm.send(level2_traffic_in, dest=comm_size-1, tag=2)
