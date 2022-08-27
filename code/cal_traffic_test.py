import numpy as np
import time
import os
import torch
from generate_map import GenerateMap
from parallelism import Parallelism
from map_analysis import MapAnalysisParallel
import copy


# 用于测试map表流量，即计算指定gpu的输出流量

class CalculateTrafficTest(MapAnalysisParallel):
    def __init__(self):
        super().__init__()
        self.output_sum_input = None
        self.output_sum_output = None
        self.pop_traffic_dict = {}
        self.route_dict_table_file = 'route_default_2dim_100_100.npy'
        self.route_dict_table = np.load(self.route_dict_table_file)

        # map的npy是一个dict文件，要加.item
        # self.map_table_split_file = '/public/home/ssct005t/project/wml_istbi/tables/map_table/map_10000/map_table_split.npy'
        # self.map_table_split = np.load(self.map_table_split_file,
        #                                allow_pickle=True).item()

        # 这个map得到的即是字典类型,如果key是str,或者是列表类型，则需要相应的转换
        self.map_table_split_file = "/public/home/ssct005t/project/wml_istbi/tables/map_table/map_2000/82experimentByInput/020000_map_table_split.npy"
        self.map_table_split = np.load(self.map_table_split_file,
                                       allow_pickle=True).item()

        if isinstance(self.map_table_split[0], list):
            self.map_table_split = self.read_map_dict_pkl(self.map_table_split_file)
        else:
            # pass
            map_table_split_int = {}
            for k, v in self.map_table_split.items():
                map_table_split_int[int(k)] = v
            self.map_table_split = map_table_split_int

    def compute_2_dim_traffic_for_sepcific_gpu_list(self, idx_list):
        self.route_dict_table_file = '/public/home/ssct005t/project/wml_istbi/code/generate_route/route_default_2dim_40_50.npy'
        self.route_dict_table = np.load(self.route_dict_table_file)

        if self.rank == self.master_rank:
            time_start = time.time()
            traffic_table_base_gpu_out_in = np.zeros((self.N, 4))
            for col_idx in range(self.N):
                time1 = time.time()
                msg_out_in = self.comm.recv(source=(col_idx % (self.comm_size - 1)))
                traffic_table_base_gpu_out_in += msg_out_in

                time2 = time.time()
                # print('Col %d: %.4fs consumed.' % (col_idx, time2 - time1))
            time_end = time.time()
            print("%d nodes used. " % ((self.comm_size - 1) / 8 + 1))
            print("%.f seconds consumed." % (time_end - time_start))
            print("计算得到流量")
            self.show_specific_gpu_traffic_info(traffic_table_base_gpu_out_in, idx_list)
        else:
            column_idx_to_process = self.allocate_idx_to_calculate()
            for gpu_out_idx in column_idx_to_process:
                traffic_table_base_gpu_out_tmp = np.zeros((self.N, 2))
                traffic_table_base_gpu_in_tmp = np.zeros((self.N, 2))
                time1 = time.time()
                traffic_table_base_gpu_out_tmp, traffic_table_base_gpu_in_tmp = self.calculate_2_dim_input_output_traffic_for_sepcific_gpu_list(
                    gpu_out_idx, idx_list)
                time2 = time.time()
                traffic_table_base_gpu_out_in = np.concatenate(
                    (traffic_table_base_gpu_out_tmp, traffic_table_base_gpu_in_tmp), axis=1)
                self.comm.send(traffic_table_base_gpu_out_in, dest=self.master_rank)
                print('Col %d sent: %.4fs consumed.' % (gpu_out_idx, time2 - time1))

    def is_changed_gpu(self, input_output_idx, specific_idx_list):
        for i in input_output_idx:
            for j in specific_idx_list:
                if i == j:
                    return True
        return False

    def calculate_2_dim_input_output_traffic_for_sepcific_gpu_list(self, idx, idx_list):
        self.dimensions = 2
        tem_output_traffic = np.zeros((self.N, self.dimensions))
        tem_input_traffic = np.zeros((self.N, self.dimensions))
        route_dict_out_idx = self.get_route_dict_out(idx, range(self.N))
        for in_idx, in_idx_list in route_dict_out_idx.items():
            stage = 0
            if len(in_idx_list) == 1:
                if not self.is_in_same_node([idx, in_idx]):
                    if self.is_changed_gpu([idx, in_idx], idx_list):
                        tmp = self.compute_2_dim_traffic_between_two_gpu(idx, in_idx)
                        tem_output_traffic[idx][stage] += tmp
                        tem_input_traffic[in_idx][stage] += tmp
            else:
                in_idx_list_tmp = list(in_idx_list)[:]
                in_idx_list_tmp.append(idx)
                in_idx_list_tmp.append(in_idx)
                if self.is_changed_gpu(in_idx_list_tmp, idx_list):
                    tmp = self.compute_2_dim_traffic_between_gpu_and_group(idx, in_idx_list)
                    tem_output_traffic[idx][stage] += tmp
                    tem_input_traffic[in_idx][stage] += tmp
                route_dict_out_idx_tmp_1 = self.get_route_dict_out(in_idx, in_idx_list)
                for in_idx_1, in_idx_list_1 in route_dict_out_idx_tmp_1.items():
                    if not self.is_in_same_node([idx, in_idx_1]):
                        if self.is_changed_gpu([idx, in_idx, in_idx_1], idx_list):
                            stage = 1
                            tmp = self.compute_2_dim_traffic_between_two_gpu(idx, in_idx_1)
                            tem_output_traffic[int(in_idx)][stage] += tmp
                            tem_input_traffic[int(in_idx_1)][stage] += tmp
        return tem_output_traffic, tem_input_traffic

        # 当以2级虚拟拓扑进行通信时，每个进程1/2级的发送/接收流量之和

    def show_specific_gpu_traffic_info(self, traffic_table_base_gpu_out_in, idx_list):
        for i in idx_list:
            print("specific: " + str(i) + " output traffic: " + str(
                traffic_table_base_gpu_out_in[i][0] + traffic_table_base_gpu_out_in[i][1]))

            print("specific: " + str(i) + " input traffic: " + str(
                traffic_table_base_gpu_out_in[i][2] + traffic_table_base_gpu_out_in[i][3]))


if __name__ == "__main__":
    g = CalculateTrafficTest()

    # traffic_table = np.load(
    #     "/public/home/ssct005t/project/wml_istbi/tables/traffic_table/map_10000_1.05_cortical_v2_without_invalid_idx/route_default_2dim_100_100/traffic_table_base_dcu_out_in_2_dim20000_sequential.npy")
    # traffic_table_t = traffic_table.T
    # output_sum = traffic_table_t[0] + traffic_table_t[1]
    #
    # input_sum = traffic_table_t[2] + traffic_table_t[3]
    # step_sort_index = output_sum.argsort()
    # idx_list = []
    # # 得到最大和最小的五个
    # for i in range(5):
    #     idx_list.append(step_sort_index[i])
    #
    # for i in range(5):
    #     idx_list.append(step_sort_index[-i])
    #
    # if g.rank == g.master_rank:
    #     print("实际流量")
    #     g.show_specific_gpu_traffic_info(traffic_table, idx_list)
    idx_list = []
    start_max = (1978 // 4) * 4
    start_min = (1759 // 4) * 4

    for i in range(4):
        idx_list.append(start_max + i)
        idx_list.append(start_min + i)
    idx_list.append(1999)
    idx_list.append(285)

    g.compute_2_dim_traffic_for_sepcific_gpu_list(idx_list)
