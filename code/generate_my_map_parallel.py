import numpy as np
import time
import os
import torch
from generate_map import GenerateMap
from parallelism import Parallelism
from map_analysis import MapAnalysisParallel
import copy


class GenerateMyMapParallel(MapAnalysisParallel):
    def __init__(self):
        super().__init__()
        self.output_sum_input = None
        self.output_sum_output = None
        self.pop_traffic_dict = {}
        self.route_dict_table_file = '/public/home/ssct005t/project/wml_istbi/code/generate_route/route_default_2dim_40_50.npy'
        self.route_dict_table = np.load(self.route_dict_table_file)

        # map的npy是一个dict文件，要加.item
        # self.map_table_split_file = '/public/home/ssct005t/project/wml_istbi/tables/map_table/map_10000/map_table_split.npy'
        # self.map_table_split = np.load(self.map_table_split_file,
        #                                allow_pickle=True).item()

        # 这个map得到的即是字典类型
        # self.map_table_split_file = "/public/home/ssct005t/project/wml_istbi/tables/map_table/map_2000/82experimentByInput/2120000_map_table_split.npy"
        # self.map_table_split = np.load(self.map_table_split_file,
        #                                allow_pickle=True).item()

        # 2000卡最初的map表
        self.map_table_split_file = "/public/home/ssct005t/project/wml_istbi/code/data_test/map_2000_dict.npy"
        self.map_table_split = np.load(self.map_table_split_file,
                                       allow_pickle=True).item()

        # self.map_table_split = np.load(self.map_table_split_file,
        #                                allow_pickle=True)

        self.map_path = "/public/home/ssct005t/project/wml_istbi/tables/map_table/"

        # 自定义每次迭代的map保存路径
        self.map_path = self.map_path + "map_2000/83experimentByInput/"
        if not os.path.exists(self.map_path):
            os.makedirs(self.map_path)

        if isinstance(self.map_table_split['0'], list):
            self.map_table_split = self.read_map_dict_pkl(self.map_table_split_file)
        else:
            # pass
            map_table_split_int = {}
            for k, v in self.map_table_split.items():
                map_table_split_int[int(k)] = v
            self.map_table_split = map_table_split_int
        self.output_sum = None
        self.traffic_table = None

    def compute_2_dim_traffic_between_two_gpu_with_pop_traffic(self, gpu_out_idx, gpu_in_idx):
        dcu_name = "cuda:" + str(self.rank % 4)
        device = torch.device(dcu_name if torch.cuda.is_available() else "cpu")

        traffic_gpu_to_gpu = 0
        pop_to_gpu_dict = {}
        for population_out_idx in self.map_table_split[gpu_out_idx].keys():
            traffic_src_to_dst = list()
            for population_in_idx in self.map_table_split[gpu_in_idx].keys():
                tmp = int(population_in_idx * self.n + population_out_idx)
                if tmp in self.conn_dict:
                    conn_number_estimate = self.neuron_number * self.size[population_in_idx] * \
                                           self.map_table_split[gpu_in_idx][population_in_idx] * self.degree[
                                               population_in_idx] * self.conn_dict[tmp]
                    traffic_src_to_dst.append(conn_number_estimate)
                else:
                    traffic_src_to_dst.append(0.0)

            sample_range = int(self.neuron_number * self.size[population_out_idx] * self.map_table_split[gpu_out_idx][
                population_out_idx])
            sample_times = int(np.sum(traffic_src_to_dst))

            torch.cuda.empty_cache()
            n = 1
            if sample_range > 1e7 or sample_times > 1e8:
                n = 30
            traffic_src_to_gpu = self.sample(sample_range, sample_times, n_slice=n)
            torch.cuda.empty_cache()
            key = str(population_out_idx) + str(self.map_table_split[gpu_out_idx][
                                                    population_out_idx]) + str(" ") + str(gpu_in_idx)
            pop_to_gpu_dict[key] = traffic_src_to_gpu
            traffic_gpu_to_gpu += traffic_src_to_gpu
        self.pop_traffic_dict.update(pop_to_gpu_dict)
        return traffic_gpu_to_gpu

    def compute_2_dim_traffic_between_gpu_and_group_with_pop_traffic(self, gpu_out_idx, gpu_in_idx):
        dcu_name = "cuda:" + str(self.rank % 4)
        device = torch.device(dcu_name if torch.cuda.is_available() else "cpu")

        traffic_gpu_to_gpu = 0
        pop_to_gpu_dict = {}
        for population_out_idx in self.map_table_split[gpu_out_idx].keys():
            traffic_src_to_dst = list()
            # gpu_in_idx_arr = self.get_idx_array(gpu_in_idx)
            # st = gpu_in_idx * self.n_gpu_per_group
            # en = st + self.n_gpu_per_group
            in_idx_str = ""
            for in_idx in gpu_in_idx:
                in_idx_str = in_idx_str + str(in_idx)
                for population_in_idx in self.map_table_split[int(in_idx)].keys():
                    tmp = int(population_in_idx * self.n + population_out_idx)
                    if tmp in self.conn_dict:
                        conn_number_estimate = self.neuron_number * self.size[population_in_idx] * \
                                               self.map_table_split[in_idx][population_in_idx] * self.degree[
                                                   population_in_idx] * self.conn_dict[tmp]
                        traffic_src_to_dst.append(conn_number_estimate)
                    else:
                        traffic_src_to_dst.append(0.0)

            sample_range = int(self.neuron_number * self.size[population_out_idx] * self.map_table_split[gpu_out_idx][
                population_out_idx])
            sample_times = int(np.sum(traffic_src_to_dst))
            # print()
            # print("Sample range: %d, Sample times: %d" % (sample_range, sample_times))

            torch.cuda.empty_cache()
            n = 1
            if sample_range > 1e7 or sample_times > 1e8:
                n = 30
            traffic_src_to_gpu = self.sample(sample_range, sample_times, n_slice=n)
            torch.cuda.empty_cache()

            key = str(population_out_idx) + str(self.map_table_split[gpu_out_idx][
                                                    population_out_idx]) + str(" ") + in_idx_str
            pop_to_gpu_dict[key] = traffic_src_to_gpu
            traffic_gpu_to_gpu += traffic_src_to_gpu
        self.pop_traffic_dict.update(pop_to_gpu_dict)
        # output.append(traffic_src_to_dcu)
        return traffic_gpu_to_gpu

    def calculate_2_dim_input_output_traffic_with_pop_traffic(self, idx):
        self.dimensions = 2
        tem_output_traffic = np.zeros((self.N, self.dimensions))
        tem_input_traffic = np.zeros((self.N, self.dimensions))
        route_dict_out_idx = self.get_route_dict_out(idx, range(self.N))
        for in_idx, in_idx_list in route_dict_out_idx.items():
            stage = 0
            if len(in_idx_list) == 1:
                tmp = self.compute_2_dim_traffic_between_two_gpu_with_pop_traffic(idx, in_idx)
                tem_output_traffic[idx][stage] += tmp
                tem_input_traffic[in_idx][stage] += tmp
            else:
                tmp = self.compute_2_dim_traffic_between_gpu_and_group_with_pop_traffic(idx, in_idx_list)
                tem_output_traffic[idx][stage] += tmp
                tem_input_traffic[in_idx][stage] += tmp
                route_dict_out_idx_tmp_1 = self.get_route_dict_out(in_idx, in_idx_list)
                for in_idx_1, in_idx_list_1 in route_dict_out_idx_tmp_1.items():
                    stage = 1
                    tmp = self.compute_2_dim_traffic_between_two_gpu_with_pop_traffic(idx, in_idx_1)
                    tem_output_traffic[int(in_idx)][stage] += tmp
                    tem_input_traffic[int(in_idx_1)][stage] += tmp

        return tem_output_traffic, tem_input_traffic

    def compute_2_dim_traffic_with_pop_traffic(self):
        self.route_dict_table_file = 'route_default_2dim_100_100.npy'
        self.route_dict_table = np.load(self.route_dict_table_file)

        if self.rank == self.master_rank:
            time_start = time.time()
            traffic_table_base_gpu_out_in = np.zeros((self.N, 4))
            # traffic_table_base_gpu_in = np.zeros((self.N, self.N))
            pop_traffic_dict_tmp = {}

            for col_idx in range(self.N):
                time1 = time.time()
                msg_out_in = self.comm.recv(source=(col_idx % (self.comm_size - 1)))
                msg_traffic_dict = self.comm.recv(source=(col_idx % (self.comm_size - 1)))
                traffic_table_base_gpu_out_in += msg_out_in
                self.pop_traffic_dict.update(msg_traffic_dict)
                time2 = time.time()
                print('Col %d: %.4fs consumed.' % (col_idx, time2 - time1))

            time.sleep(10)

            np.save("pop_to_gpu_traffic.npy", self.pop_traffic_dict)

            # 建立二维的路径
            file_path = self.traffic_table_root + self.map_table_split_file[0:len(
                self.map_table_split_file) - 4] + '/' + self.route_dict_table_file[
                                                        0:len(self.route_dict_table_file) - 4] + '/'
            if not os.path.exists(file_path):
                os.makedirs(file_path)

            npy_name = "traffic_table_base_dcu_out_in_2_dim" + self.map_version + ".npy"
            np.save(file_path + npy_name, traffic_table_base_gpu_out_in)
            print(file_path + npy_name + " saved.")
            traffic_table_base_gpu_out_in_t = traffic_table_base_gpu_out_in.T
            for i in range(4):
                if i < 2:
                    figure_name = file_path + 'traffic_table_out_' + str(i) + "stage" + '.png'
                else:
                    figure_name = file_path + 'traffic_table_in_' + str(i - 2) + "stage" + '.png'
                self.draw(traffic_table_base_gpu_out_in_t[i], figure_name)

            figure_name = file_path + "traffic_table_out_sum.png"
            self.draw(traffic_table_base_gpu_out_in_t[0] + traffic_table_base_gpu_out_in_t[1], figure_name)
            figure_name = file_path + "traffic_table_in_sum.png"
            self.draw(traffic_table_base_gpu_out_in_t[2] + traffic_table_base_gpu_out_in_t[3], figure_name)

            # npy_name = "traffic_table_base_dcu_in" + self.map_version + ".npy"
            # np.save(self.traffic_table_root + npy_name, traffic_table_base_gpu_in)
            # print(self.traffic_table_root + npy_name + " saved.")

            time_end = time.time()
            print("%d nodes used. " % ((self.comm_size - 1) / 8 + 1))
            print("%.f seconds consumed." % (time_end - time_start))


        else:
            column_idx_to_process = self.allocate_idx_to_calculate()
            for gpu_out_idx in column_idx_to_process:
                # calculate the traffic, stored as a (N, 1) array
                # traffic_base_gpu_for_a_column_out = np.zeros((self.N,))
                traffic_table_base_gpu_out_tmp = np.zeros((self.N, 2))
                traffic_table_base_gpu_in_tmp = np.zeros((self.N, 2))
                # traffic_base_gpu_for_a_column_in = np.zeros((self.N,))
                time1 = time.time()
                traffic_table_base_gpu_out_tmp, traffic_table_base_gpu_in_tmp = self.calculate_2_dim_input_output_traffic_with_pop_traffic(
                    gpu_out_idx)
                time2 = time.time()
                traffic_table_base_gpu_out_in = np.concatenate(
                    (traffic_table_base_gpu_out_tmp, traffic_table_base_gpu_in_tmp), axis=1)
                self.comm.send(traffic_table_base_gpu_out_in, dest=self.master_rank)
                self.comm.send(self.pop_traffic_dict, dest=self.master_rank)
                # self.comm.send(traffic_base_gpu_for_a_column_in, dest=self.master_rank)
                print('Col %d sent: %.4fs consumed.' % (gpu_out_idx, time2 - time1))

    def calculate_2_dim_output_traffic_for_map_out_in(self, idx):
        self.dimensions = 2
        tem_output_traffic = np.zeros((self.N,))
        tem_input_traffic = np.zeros((self.N,))

        route_dict_out_idx = self.get_route_dict_out(idx, range(self.N))

        for in_idx, in_idx_list in route_dict_out_idx.items():
            if len(in_idx_list) == 1:
                if self.is_changed_gpu_traffic([in_idx, idx]):
                    if not self.is_in_same_node([idx, in_idx]):  # 不在同一个节点内(节点内通过PIC通信，不考虑)
                        tmp1 = self.compute_2_dim_traffic_between_two_gpu(idx, in_idx)
                        tmp2 = self.compute_2_dim_traffic_between_two_gpu_old(idx, in_idx)
                        tem_output_traffic[idx] += (tmp1 - tmp2)
                        tem_input_traffic[in_idx] += (tmp1 - tmp2)
            else:
                in_idx_list_tmp = list(in_idx_list)[:]
                in_idx_list_tmp.append(idx)
                in_idx_list_tmp.append(in_idx)
                if self.is_changed_gpu_traffic(in_idx_list_tmp):
                    tmp1 = self.compute_2_dim_traffic_between_gpu_and_group(idx, in_idx_list)
                    tmp2 = self.compute_2_dim_traffic_between_gpu_and_group_old(idx, in_idx_list)
                    tem_output_traffic[idx] += (tmp1 - tmp2)
                    tem_input_traffic[in_idx] += (tmp1 - tmp2)
                route_dict_out_idx_tmp_1 = self.get_route_dict_out(in_idx, in_idx_list)
                for in_idx_1, in_idx_list_1 in route_dict_out_idx_tmp_1.items():
                    if self.is_changed_gpu_traffic([in_idx_1, in_idx, idx]):
                        if not self.is_in_same_node([idx, in_idx_1]):
                            tmp1 = self.compute_2_dim_traffic_between_two_gpu(idx, in_idx_1)
                            tmp2 = self.compute_2_dim_traffic_between_two_gpu_old(idx, in_idx_1)
                            tem_output_traffic[int(in_idx)] += (tmp1 - tmp2)
                            tem_input_traffic[int(in_idx_1)] += (tmp1 - tmp2)
        return tem_output_traffic, tem_input_traffic

    # 按照输入和输出的加和进行迭代，效果不好
    def generate_map_by_input_output(self):
        if self.rank == self.master_rank:
            iter_count = 0
            iter_sum = 70
            max_traffic_iter = np.zeros((iter_sum,))
            average_traffic_iter = np.zeros((iter_sum,))
            min_traffic_iter = np.zeros((iter_sum,))
            while iter_count < iter_sum:
                time1 = time.time()
                if iter_count == 0:
                    self.traffic_table = np.load(
                        "traffic_table_base_dcu_out_in_2_dim20000_sequential.npy",
                        allow_pickle=True)
                    traffic_table_t = self.traffic_table.T
                    self.output_sum = traffic_table_t[0] + traffic_table_t[1] + traffic_table_t[2] + traffic_table_t[3]

                    self.output_sum_output = traffic_table_t[0] + traffic_table_t[1]
                    self.output_sum_input = traffic_table_t[2] + traffic_table_t[3]

                    print("output 19810 info")
                    print("traffic " + str(self.output_sum_output[19810]))
                    print("input 19810 info")
                    print("traffic " + str(self.output_sum_input[19810]))

                    print("sum traffic 19810 info")
                    print("traffic " + str(self.output_sum[19810]))

                    print("output sum info:  ------------")
                    print("average: " + str(np.average(self.output_sum_output)))
                    print("max: " + str(np.max(self.output_sum_output)))
                    print("min: " + str(np.min(self.output_sum_output)))

                    print("input sum info:  ------------")
                    print("average: " + str(np.average(self.output_sum_input)))
                    print("max: " + str(np.max(self.output_sum_input)))
                    print("min: " + str(np.min(self.output_sum_input)))

                    self.traffic_table = self.output_sum

                    print("sum info :  ------------")
                    print("average: " + str(np.average(self.output_sum)))
                    print("max: " + str(np.max(self.output_sum)))
                    print("min: " + str(np.min(self.output_sum)))

                self.split_per_pop_by_swap_min_max()
                # self.split_per_pop_by_step_v2()
                self.map_table_split = self.comm.bcast(self.map_table_split, self.master_rank)
                self.map_table_split_old = self.comm.bcast(self.map_table_split_old, self.master_rank)
                self.gpu_changed_idx = self.comm.bcast(self.gpu_changed_idx, self.master_rank)
                self.comm.barrier()
                time_start = time.time()
                # 这里是两个阶段，也可以不分阶段优化
                traffic_table_base_gpu_out_t = np.zeros((1, self.N))
                traffic_table_base_gpu_in_t = np.zeros((1, self.N))

                for col_idx in range(self.N):
                    msg_out = self.comm.recv(source=(col_idx % (self.comm_size - 1)))

                    msg_out_t = msg_out.T

                    traffic_table_base_gpu_out_t += msg_out_t[0]
                    traffic_table_base_gpu_in_t += msg_out_t[1]

                # traffic_table_base_gpu_out_t = traffic_table_base_gpu_out.T
                # traffic_table_base_gpu_in_t = traffic_table_base_gpu_in.T
                tmp_out_traffic = traffic_table_base_gpu_out_t[0]
                tmp_in_traffic = traffic_table_base_gpu_in_t[0]

                print("output 19810 info")
                print("traffic " + str(tmp_out_traffic[19810]))
                print("input 19810 info")
                print("traffic " + str(tmp_in_traffic[19810]))

                traffic_table_base_gpu_out_in_t = traffic_table_base_gpu_in_t + traffic_table_base_gpu_out_t
                tmp_traffic = traffic_table_base_gpu_out_in_t[0]

                print("sum trarffic 19810 info")
                print("traffic " + str(tmp_out_traffic[19810]))

                print("the div:  ------------")
                print("average: " + str(np.average(tmp_traffic)))
                print("max: " + str(np.max(tmp_traffic)) + " " + str(np.argmax(tmp_traffic)))
                print("min: " + str(np.min(tmp_traffic)) + " " + str(np.argmin(tmp_traffic)))
                self.output_sum = self.output_sum + tmp_traffic
                self.traffic_table = self.output_sum

                print("********")
                for i in self.gpu_changed_idx:
                    print(str(i) + " " + str(tmp_traffic[i]) + " " + str(self.output_sum[i]))
                print("********")
                max_traffic_iter[iter_count] = np.max(self.output_sum)
                min_traffic_iter[iter_count] = np.min(self.output_sum)
                average_traffic_iter[iter_count] = np.average(self.output_sum)

                print("output sum info:  ------------")
                self.output_sum_output = self.output_sum_output + tmp_out_traffic
                print("average: " + str(np.average(self.output_sum_output)))
                print("max: " + str(np.max(self.output_sum_output)))
                print("min: " + str(np.min(self.output_sum_output)))

                print("input sum info:  ------------")
                self.output_sum_input = self.output_sum_input + tmp_in_traffic
                print("average: " + str(np.average(self.output_sum_input)))
                print("max: " + str(np.max(self.output_sum_input)))
                print("min: " + str(np.min(self.output_sum_input)))

                print("the sum output:  ------------")
                print("average: " + str(np.average(self.output_sum)))
                print("max: " + str(np.max(self.output_sum)))
                print("min: " + str(np.min(self.output_sum)))
                time2 = time.time()
                print("one iteration consumed {0} seconds".format(time2 - time1))
                map_table_name = str(iter_count) + "20000_map_table_split.npy"
                np.save(map_table_name, self.map_table_split)

                print(map_table_name + " saved")
                iter_count += 1
            print(max_traffic_iter)
            print(average_traffic_iter)
            print(min_traffic_iter)
            np.save("output_max.npy", max_traffic_iter)
            np.save("output_average.npy", average_traffic_iter)
            np.save("output_min.npy", min_traffic_iter)
        else:
            iter_count = 0
            while iter_count < 70:
                column_idx_to_process = self.allocate_idx_to_calculate()
                self.map_table_split = self.comm.bcast(self.map_table_split, self.master_rank)
                self.map_table_split_old = self.comm.bcast(self.map_table_split_old, self.master_rank)
                self.gpu_changed_idx = self.comm.bcast(self.gpu_changed_idx, self.master_rank)
                self.comm.barrier()
                for gpu_out_idx in column_idx_to_process:
                    traffic_table_base_gpu_out_tmp = np.zeros((self.N, 1))
                    traffic_table_base_gpu_in_tmp = np.zeros((self.N, 1))
                    time1 = time.time()
                    traffic_table_base_gpu_out_tmp, traffic_table_base_gpu_in_tmp = self.calculate_2_dim_output_traffic_for_map_out_in(
                        gpu_out_idx)
                    traffic_table_base_gpu_out_in = np.concatenate(
                        (traffic_table_base_gpu_out_tmp, traffic_table_base_gpu_in_tmp), axis=1)

                    time2 = time.time()
                    self.comm.send(traffic_table_base_gpu_out_in, dest=self.master_rank)
                    # self.comm.send(traffic_base_gpu_for_a_column_in, dest=self.master_rank)
                    if gpu_out_idx % 1000 == 0:
                        print('Col %d sent: %.4fs consumed.' % (gpu_out_idx, time2 - time1))
                iter_count += 1

    def generate_map_by_output_or_input(self):
        if self.rank == self.master_rank:
            iter_count = 0
            iter_sum = 70
            max_traffic_iter = np.zeros((iter_sum,))
            average_traffic_iter = np.zeros((iter_sum,))
            min_traffic_iter = np.zeros((iter_sum,))
            while iter_count < iter_sum:
                time1 = time.time()
                if iter_count == 0:
                    # self.traffic_table = np.load(
                    #     "/public/home/ssct005t/project/wml_istbi/tables/traffic_table/public/home/ssct005t/project/wml_istbi/tables/map_table/map_10000/map_table_split/route_default_2dim_100_100/traffic_table_base_dcu_out_in_2_dim20000_sequential.npy",
                    #     allow_pickle=True)
                    self.traffic_table = np.load(
                        "/public/home/ssct005t/project/wml_istbi/tables/traffic_table/public/home/ssct005t/project/wml_istbi/code/data_test/map_2000_dict/public/home/ssct005t/project/wml_istbi/code/generate_route/route_default_2dim_40_50/traffic_table_base_dcu_out_in_2_dim20000_sequential.npy",
                        allow_pickle=True)

                    traffic_table_t = self.traffic_table.T
                    self.output_sum = traffic_table_t[0] + traffic_table_t[1] + traffic_table_t[2] + traffic_table_t[3]
                    self.output_sum_output = traffic_table_t[0] + traffic_table_t[1]
                    self.output_sum_input = traffic_table_t[2] + traffic_table_t[3]
                    self.traffic_table = self.output_sum

                    self.show_traffic_info()

                # 按照out_sum_output也就是输出流量迭代，这里可以直接替换
                # self.split_per_pop_by_swap_min_max(self.output_sum_input)
                print("拆分")
                # self.split_pop_by_step(self.output_sum_output)
                # self.split_per_pop_by_step(self.output_sum_output)
                # self.split_per_pop_by_step_v2()
                # self.split_pop_test(self.output_sum_output)

                traffic_output_input_max = np.zeros((self.N, 1))
                for i in range(self.N):
                    traffic_output_input_max[i] = max(self.output_sum_input[i], self.output_sum_output[i])

                self.split_per_pop_by_swap_min_max(traffic_output_input_max)

                self.map_table_split = self.comm.bcast(self.map_table_split, self.master_rank)
                self.map_table_split_old = self.comm.bcast(self.map_table_split_old, self.master_rank)
                self.gpu_changed_idx = self.comm.bcast(self.gpu_changed_idx, self.master_rank)
                self.comm.barrier()
                time_start = time.time()
                # 这里是两个阶段，也可以不分阶段优化,但是两个矩阵结合成一个更新会出现数值过小的问题
                traffic_table_base_gpu_out_t = np.zeros((1, self.N))
                traffic_table_base_gpu_in_t = np.zeros((1, self.N))

                for col_idx in range(self.N):
                    msg_out = self.comm.recv(source=(col_idx % (self.comm_size - 1)))
                    msg_out_t = msg_out.T
                    traffic_table_base_gpu_out_t += msg_out_t[0]
                    traffic_table_base_gpu_in_t += msg_out_t[1]

                # traffic_table_base_gpu_out_t = traffic_table_base_gpu_out.T
                # traffic_table_base_gpu_in_t = traffic_table_base_gpu_in.T
                tmp_out_traffic = traffic_table_base_gpu_out_t[0]
                tmp_in_traffic = traffic_table_base_gpu_in_t[0]

                traffic_table_base_gpu_out_in_t = traffic_table_base_gpu_in_t + traffic_table_base_gpu_out_t
                tmp_traffic = traffic_table_base_gpu_out_in_t[0]

                print("the div:  ------------")
                print("average: " + str(np.average(tmp_traffic)))
                print("max: " + str(np.max(tmp_traffic)) + " " + str(np.argmax(tmp_traffic)))
                print("min: " + str(np.min(tmp_traffic)) + " " + str(np.argmin(tmp_traffic)))
                self.output_sum = self.output_sum + tmp_traffic
                self.traffic_table = self.output_sum

                # print("********")
                # for i in self.gpu_changed_idx:
                #     print(str(i) + " " + str(tmp_traffic[i]) + " " + str(self.output_sum[i]))
                # print("********")
                max_traffic_iter[iter_count] = np.max(self.output_sum)
                min_traffic_iter[iter_count] = np.min(self.output_sum)
                average_traffic_iter[iter_count] = np.average(self.output_sum)

                self.output_sum_output = self.output_sum_output + tmp_out_traffic
                self.output_sum_input = self.output_sum_input + tmp_in_traffic

                self.show_traffic_info()

                time2 = time.time()
                print("one iteration consumed {0} seconds".format(time2 - time1))

                map_table_name = self.map_path + str(iter_count) + "20000_map_table_split.npy"
                np.save(map_table_name, self.map_table_split)
                print(map_table_name + " saved")
                iter_count += 1
            #     这里可以增加一个画图
            print(max_traffic_iter)
            print(average_traffic_iter)
            print(min_traffic_iter)
            np.save(self.map_path + "output_max.npy", max_traffic_iter)
            np.save(self.map_path + "output_average.npy", average_traffic_iter)
            np.save(self.map_path + "output_min.npy", min_traffic_iter)
        else:
            iter_count = 0
            while iter_count < 70:
                column_idx_to_process = self.allocate_idx_to_calculate()
                self.map_table_split = self.comm.bcast(self.map_table_split, self.master_rank)
                self.map_table_split_old = self.comm.bcast(self.map_table_split_old, self.master_rank)
                self.gpu_changed_idx = self.comm.bcast(self.gpu_changed_idx, self.master_rank)
                self.comm.barrier()
                for gpu_out_idx in column_idx_to_process:
                    traffic_table_base_gpu_out_in = np.zeros((self.N, 2))
                    traffic_table_base_gpu_out_tmp = np.zeros((self.N, 1))
                    traffic_table_base_gpu_in_tmp = np.zeros((self.N, 1))
                    time1 = time.time()
                    # traffic_table_base_gpu_out_tmp, traffic_table_base_gpu_in_tmp = self.calculate_2_dim_output_traffic_for_map_out_in(
                    #     gpu_out_idx)

                    traffic_table_base_gpu_out_tmp, traffic_table_base_gpu_in_tmp = self.calculate_2_dim_output_traffic_for_map_out_in_optimize_input_calculations(
                        gpu_out_idx)

                    for i in range(self.N):
                        traffic_table_base_gpu_out_in[i][0] = traffic_table_base_gpu_out_tmp[i]
                        traffic_table_base_gpu_out_in[i][1] = traffic_table_base_gpu_out_tmp[i]
                    # traffic_table_base_gpu_out_in = np.concatenate(
                    #     (traffic_table_base_gpu_out_tmp, traffic_table_base_gpu_in_tmp), axis=1)

                    time2 = time.time()
                    self.comm.send(traffic_table_base_gpu_out_in, dest=self.master_rank)
                    # self.comm.send(traffic_base_gpu_for_a_column_in, dest=self.master_rank)
                    if gpu_out_idx % 1000 == 0:
                        print('Col %d sent: %.4fs consumed.' % (gpu_out_idx, time2 - time1))
                iter_count += 1

    def show_traffic_info(self):
        print("output sum info:  ------------")
        print("average: " + str(np.average(self.output_sum_output)))
        print("max: " + str(np.max(self.output_sum_output)))
        print("min: " + str(np.min(self.output_sum_output)))

        print("input sum info:  ------------")
        print("average: " + str(np.average(self.output_sum_input)))
        print("max: " + str(np.max(self.output_sum_input)))
        print("min: " + str(np.min(self.output_sum_input)))

        print("sum info :  ------------")
        print("average: " + str(np.average(self.output_sum)))
        print("max: " + str(np.max(self.output_sum)))
        print("min: " + str(np.min(self.output_sum)))

        self.step_sort_index = self.output_sum_output.argsort()

        max_traffic_gpu_idx = self.step_sort_index[self.output_sum_output.shape[0] - 1]
        min_traffic_gpu_idx = self.step_sort_index[0]

        print("max gpu idx")
        print(max_traffic_gpu_idx)
        print(self.output_sum_output[max_traffic_gpu_idx])

        print("min gpu idx")
        print(min_traffic_gpu_idx)
        print(self.output_sum_output[min_traffic_gpu_idx])

        # print("2853 traffic")
        # print(self.output_sum_output[2853])
        #
        # print("1350 traffic")
        # print(self.output_sum_output[1350])

    def split_per_pop_by_swap_min_max(self, traffic_iter):
        self.map_table_split_old = np.empty((2000,), dtype=dict)
        for i in range(self.N):
            self.map_table_split_old[i] = copy.deepcopy(self.map_table_split[i])
        self.step_sort_index = traffic_iter.argsort()
        self.gpu_changed_idx = []
        for i in range(1, 10):
            max_traffic_gpu_idx = self.step_sort_index[traffic_iter.shape[0] - i]
            print(max_traffic_gpu_idx)
        idx = 0
        cnt = 0
        while cnt < self.step:
            # for i in range(self.step):
            max_traffic_gpu_idx = self.step_sort_index[traffic_iter.shape[0] - idx - 1]
            # 最小的gpu idx
            min_traffic_gpu_idx = self.step_sort_index[idx]
            idx += 1
            # 拆分的本次的gpu的pop
            flag = 0
            max_gpu_pop_size = []
            for k, v in self.map_table_split[max_traffic_gpu_idx].items():
                if v == 1:
                    max_gpu_pop_size.append(self.size[k])
                    flag = 1
            if len(max_gpu_pop_size) == 0:
                print("max_traffic_gpu_idx " + str(max_traffic_gpu_idx))
                print(self.map_table_split[max_traffic_gpu_idx])
                print("traffic: " + str(traffic_iter[max_traffic_gpu_idx]))
                continue
            cnt += 1
            max_gpu_pop_size = np.array(max_gpu_pop_size)
            max_gpu_pop_size = np.sort(max_gpu_pop_size, axis=0)
            middle_size = max_gpu_pop_size[max_gpu_pop_size.shape[0] // 2]

            if flag == 1:
                pop_idx_list = []
                for k, v in self.map_table_split[max_traffic_gpu_idx].items():
                    if v == 1:
                        if self.size[k] >= middle_size:
                            pop_idx_list.append(k)
                for k in pop_idx_list:
                    self.map_table_split[max_traffic_gpu_idx].pop(k)
                self.gpu_changed_idx.append(max_traffic_gpu_idx)

                start_gpu_idx = 4 * int(min_traffic_gpu_idx // 4)
                end_gpu_idx = start_gpu_idx + 4
                for changed_idx in range(start_gpu_idx, end_gpu_idx):
                    self.gpu_changed_idx.append(changed_idx)
                    for k in pop_idx_list:
                        self.map_table_split[changed_idx][k] = 0.25

                if self.rank == self.master_rank:
                    print(min_traffic_gpu_idx)
                    print(start_gpu_idx)
                    print(max_traffic_gpu_idx)
                    print(end_gpu_idx)
                    print("before split")
                    for i in self.gpu_changed_idx:
                        print(self.map_table_split_old[i])
                    print("after split")
                    for i in self.gpu_changed_idx:
                        print(self.map_table_split[i])

    # 对流量的迭代进行优化，(out_idx,in_idx)中当in_idx不变时，out_idx只需要计算发生变化的pop
    def calculate_2_dim_output_traffic_for_map_out_in_optimize_input_calculations(self, idx):
        self.dimensions = 2
        tem_output_traffic = np.zeros((self.N,))
        tem_input_traffic = np.zeros((self.N,))

        route_dict_out_idx = self.get_route_dict_out(idx, range(self.N))

        for in_idx, in_idx_list in route_dict_out_idx.items():
            if len(in_idx_list) == 1:  # gpu to gpu发送
                # 发送idx发生了变化(pop增减)，但是接受in_idx未变
                if self.is_changed_gpu_traffic([idx]) and not self.is_changed_gpu_traffic([in_idx]):
                    if not self.is_in_same_node([idx, in_idx]):
                        tmp = self.calculate_gpu_to_gpu_traffic_by_out_changed(idx, [in_idx])
                        tem_output_traffic[idx] += tmp
                        tem_input_traffic[in_idx] += tmp
                else:
                    # 接受in_idx发生了变化
                    if self.is_changed_gpu_traffic([in_idx]):
                        if not self.is_in_same_node([idx, in_idx]):  # 不在同一个节点内(节点内通过PIC通信，不考虑)
                            tmp1 = self.compute_2_dim_traffic_between_two_gpu(idx, in_idx)
                            tmp2 = self.compute_2_dim_traffic_between_two_gpu_old(idx, in_idx)
                            tem_output_traffic[idx] += (tmp1 - tmp2)
                            tem_input_traffic[in_idx] += (tmp1 - tmp2)

            else:  # gpu to group发送
                in_idx_list_tmp = list(in_idx_list)[:]
                # in_idx_list_tmp.append(idx)
                # 发送idx发生了变化(pop增减)，但是接受in_idx_list未变
                if self.is_changed_gpu_traffic([idx]) and not self.is_changed_gpu_traffic(in_idx_list_tmp):
                    tmp = self.calculate_gpu_to_gpu_traffic_by_out_changed(idx, in_idx_list)
                    tem_output_traffic[idx] += tmp
                    tem_input_traffic[in_idx] += tmp
                else:
                    in_idx_list_tmp.append(in_idx)
                    if self.is_changed_gpu_traffic(in_idx_list_tmp):
                        tmp1 = self.compute_2_dim_traffic_between_gpu_and_group(idx, in_idx_list)
                        tmp2 = self.compute_2_dim_traffic_between_gpu_and_group_old(idx, in_idx_list)
                        tem_output_traffic[idx] += (tmp1 - tmp2)
                        tem_input_traffic[in_idx] += (tmp1 - tmp2)
                    route_dict_out_idx_tmp_1 = self.get_route_dict_out(in_idx, in_idx_list)
                    for in_idx_1, in_idx_list_1 in route_dict_out_idx_tmp_1.items():
                        if self.is_changed_gpu_traffic([idx]) and not self.is_changed_gpu_traffic([in_idx_1]):
                            if not self.is_in_same_node([idx, in_idx_1]):
                                tmp = self.calculate_gpu_to_gpu_traffic_by_out_changed(idx, [in_idx_1])
                                tem_output_traffic[in_idx] += tmp
                                tem_input_traffic[in_idx_1] += tmp
                        else:
                            if self.is_changed_gpu_traffic([in_idx, in_idx_1]):
                                if not self.is_in_same_node([idx, in_idx_1]):
                                    tmp1 = self.compute_2_dim_traffic_between_two_gpu(idx, in_idx_1)
                                    tmp2 = self.compute_2_dim_traffic_between_two_gpu_old(idx, in_idx_1)
                                    tem_output_traffic[in_idx] += (tmp1 - tmp2)
                                    tem_input_traffic[in_idx_1] += (tmp1 - tmp2)

        return tem_output_traffic, tem_input_traffic

    # 当(out_idx,in_idx)仅有out_idx发生变化时，计算变化的流量
    def calculate_gpu_to_gpu_traffic_by_out_changed(self, gpu_out_idx, gpu_in_idx):
        dcu_name = "cuda:" + str(self.rank % 4)
        device = torch.device(dcu_name if torch.cuda.is_available() else "cpu")

        changed_pop_in_new_split = []

        for pop_idx, pop_per in self.map_table_split[gpu_out_idx].items():
            # 没有对应的key值 或 有key但是切分的per已经发生了变化
            if not (pop_idx in self.map_table_split_old[gpu_out_idx]) or (
                    pop_idx in self.map_table_split_old[gpu_out_idx] and self.map_table_split_old[
                gpu_out_idx][pop_idx] != pop_per):
                changed_pop_in_new_split.append(pop_idx)

        changed_pop_in_old_split = []
        for pop_idx, pop_per in self.map_table_split_old[gpu_out_idx].items():
            # 没有对应的key值 或 有key但是切分的per已经发生了变化
            if not (pop_idx in self.map_table_split[gpu_out_idx]) or (
                    pop_idx in self.map_table_split[gpu_out_idx] and self.map_table_split[
                gpu_out_idx][pop_idx] != pop_per):
                changed_pop_in_old_split.append(pop_idx)

        traffic_gpu_to_gpu_new = self.get_changed_pop_traffic_new(gpu_out_idx, gpu_in_idx,
                                                                  changed_pop_in_new_split)

        traffic_gpu_to_gpu_old = self.get_changed_pop_traffic_old(gpu_out_idx, gpu_in_idx, changed_pop_in_old_split)

        return traffic_gpu_to_gpu_new - traffic_gpu_to_gpu_old

    def get_changed_pop_traffic_new(self, gpu_out_idx, gpu_in_idx_list, changed_pop):
        traffic_gpu_to_gpu = 0
        for population_out_idx in changed_pop:
            traffic_src_to_dst = list()
            # 组内流量是存在并包的
            for gpu_in_idx in gpu_in_idx_list:
                for population_in_idx in self.map_table_split[gpu_in_idx].keys():
                    tmp = int(population_in_idx * self.n + population_out_idx)
                    if tmp in self.conn_dict:
                        conn_number_estimate = self.neuron_number * self.size[population_in_idx] * \
                                               self.map_table_split[gpu_in_idx][population_in_idx] * self.degree[
                                                   population_in_idx] * self.conn_dict[tmp]
                        traffic_src_to_dst.append(conn_number_estimate)
                    else:
                        traffic_src_to_dst.append(0.0)

            sample_range = int(self.neuron_number * self.size[population_out_idx] * self.map_table_split[gpu_out_idx][
                population_out_idx])
            sample_times = int(np.sum(traffic_src_to_dst))

            torch.cuda.empty_cache()
            n = 1
            if sample_range > 1e7 or sample_times > 1e8:
                n = 50
            traffic_src_to_gpu = self.sample(sample_range, sample_times, n_slice=n)
            torch.cuda.empty_cache()
            traffic_gpu_to_gpu += traffic_src_to_gpu
        return traffic_gpu_to_gpu

    def get_changed_pop_traffic_old(self, gpu_out_idx, gpu_in_idx_list, changed_pop):
        traffic_gpu_to_gpu = 0
        for population_out_idx in changed_pop:
            traffic_src_to_dst = list()
            # 组内流量是存在并包的
            for gpu_in_idx in gpu_in_idx_list:
                for population_in_idx in self.map_table_split_old[gpu_in_idx].keys():
                    tmp = int(population_in_idx * self.n + population_out_idx)
                    if tmp in self.conn_dict:
                        conn_number_estimate = self.neuron_number * self.size[population_in_idx] * \
                                               self.map_table_split_old[gpu_in_idx][population_in_idx] * self.degree[
                                                   population_in_idx] * self.conn_dict[tmp]
                        traffic_src_to_dst.append(conn_number_estimate)
                    else:
                        traffic_src_to_dst.append(0.0)

            sample_range = int(
                self.neuron_number * self.size[population_out_idx] * self.map_table_split_old[gpu_out_idx][
                    population_out_idx])
            sample_times = int(np.sum(traffic_src_to_dst))

            torch.cuda.empty_cache()
            n = 1
            if sample_range > 1e7 or sample_times > 1e8:
                n = 50
            traffic_src_to_gpu = self.sample(sample_range, sample_times, n_slice=n)
            torch.cuda.empty_cache()
            traffic_gpu_to_gpu += traffic_src_to_gpu
        return traffic_gpu_to_gpu

    # 交换最小和最大gpu所有pop，用于验证拆分是否正确
    def split_pop_test(self, output_sum_output):
        self.map_table_split_old = np.empty((self.N,), dtype=dict)
        # Deep copy dictionary, old_split needs to be used in traffic calculation
        # don't to use shallow copy
        for i in range(self.N):
            self.map_table_split_old[i] = copy.deepcopy(self.map_table_split[i])
        self.step_sort_index = output_sum_output.argsort()

        print(self.step_sort_index)
        print(output_sum_output.shape)
        max_traffic_gpu_idx = self.step_sort_index[-1]
        min_traffic_gpu_idx = self.step_sort_index[0]
        print(max_traffic_gpu_idx)
        print(min_traffic_gpu_idx)
        start_max = (max_traffic_gpu_idx // 4) * 4
        start_min = (min_traffic_gpu_idx // 4) * 4
        self.gpu_changed_idx = []
        for i in range(4):
            temp = self.map_table_split[start_max + i]
            self.map_table_split[start_max + i] = self.map_table_split[start_min + i]
            self.map_table_split[start_min + i] = temp
            self.gpu_changed_idx.append(start_max + i)
            self.gpu_changed_idx.append(start_min + i)

        print("before swap")
        print("max gpu idx")
        print(max_traffic_gpu_idx)
        print(self.map_table_split_old[max_traffic_gpu_idx])
        print("min gpu idx")
        print(min_traffic_gpu_idx)
        print(self.map_table_split_old[min_traffic_gpu_idx])
        print("after swap")
        print("max gpu idx")
        print(max_traffic_gpu_idx)
        print(self.map_table_split[max_traffic_gpu_idx])
        print("min gpu idx")
        print(min_traffic_gpu_idx)
        print(self.map_table_split[min_traffic_gpu_idx])

        # 验证拆分是否正确
        pop_dict = {}
        for i in range(2000):
            for k, v in self.map_table_split[i].items():
                if k in pop_dict:
                    pop_dict[k] += v
                else:
                    pop_dict[k] = v
        count = 0
        for v in pop_dict.values():
            count += 1
            if v != 1:
                print("拆分错误")
        if count != 171508:
            print("拆分错误")

    def generate_map_by_output(self):
        if self.rank == self.master_rank:
            iter_count = 0
            iter_sum = 70
            max_traffic_iter = np.zeros((iter_sum,))
            average_traffic_iter = np.zeros((iter_sum,))
            min_traffic_iter = np.zeros((iter_sum,))
            while iter_count < iter_sum:
                time1 = time.time()
                if iter_count == 0:
                    self.traffic_table = np.load(
                        "/public/home/ssct005t/project/wml_istbi/tables/traffic_table/public/home/ssct005t/project/wml_istbi/code/data_test/map_2000_dict/public/home/ssct005t/project/wml_istbi/code/generate_route/route_default_2dim_40_50/traffic_table_base_dcu_out_in_2_dim20000_sequential.npy",
                        allow_pickle=True)
                    traffic_table_t = self.traffic_table.T
                    self.output_sum = traffic_table_t[0] + traffic_table_t[1]
                    self.traffic_table = self.output_sum
                    print("output sum info :  ------------")
                    print("average: " + str(np.average(self.output_sum)))
                    print("max: " + str(np.max(self.output_sum)))
                    print("min: " + str(np.min(self.output_sum)))

                # 按照out_sum_output也就是输出流量迭代，这里可以直接替换
                # self.split_per_pop_by_swap_min_max(self.output_sum_input)
                print("拆分测试")
                # self.split_pop_by_step(self.output_sum_output)
                # self.split_per_pop_by_step(self.output_sum)
                # self.split_per_pop_by_step_v2()
                self.split_per_pop_by_swap_min_max(self.output_sum)
                # self.split_pop_test(self.output_sum)

                # self.gpu_changed_idx = [1978, 1759]
                # self.map_table_split_old = self.map_table_split

                self.map_table_split = self.comm.bcast(self.map_table_split, self.master_rank)
                self.map_table_split_old = self.comm.bcast(self.map_table_split_old, self.master_rank)
                self.gpu_changed_idx = self.comm.bcast(self.gpu_changed_idx, self.master_rank)
                self.comm.barrier()
                time_start = time.time()
                # 这里是两个阶段，也可以不分阶段优化,但是两个矩阵结合成一个更新会出现数值过小的问题
                traffic_table_base_gpu_out = np.zeros((self.N,))

                for col_idx in range(self.N):
                    msg_out = self.comm.recv(source=(col_idx % (self.comm_size - 1)))
                    traffic_table_base_gpu_out += msg_out

                tmp_traffic = traffic_table_base_gpu_out
                # traffic_table_base_gpu_out_t = traffic_table_base_gpu_out.T
                # traffic_table_base_gpu_in_t = traffic_table_base_gpu_in.T

                print("the div:  ------------")
                print("average: " + str(np.average(tmp_traffic)))
                print("max: " + str(np.max(tmp_traffic)) + " " + str(np.argmax(tmp_traffic)))
                print("min: " + str(np.min(tmp_traffic)) + " " + str(np.argmin(tmp_traffic)))

                for i in self.gpu_changed_idx:
                    print("temp traffic %d " % (i))
                    print(tmp_traffic[i])

                self.output_sum = self.output_sum + tmp_traffic
                for i in self.gpu_changed_idx:
                    print("temp traffic %d " % (i))
                    print(self.output_sum[i])
                self.traffic_table = self.output_sum

                # print("********")
                # for i in self.gpu_changed_idx:
                #     print(str(i) + " " + str(tmp_traffic[i]) + " " + str(self.output_sum[i]))
                # print("********")
                max_traffic_iter[iter_count] = np.max(self.output_sum)
                min_traffic_iter[iter_count] = np.min(self.output_sum)
                average_traffic_iter[iter_count] = np.average(self.output_sum)

                print("output sum info :  ------------")
                print("average: " + str(np.average(self.output_sum)))
                print("max: " + str(np.max(self.output_sum)))
                print("min: " + str(np.min(self.output_sum)))

                time2 = time.time()
                print("one iteration consumed {0} seconds".format(time2 - time1))

                map_table_name = self.map_path + str(iter_count) + "20000_map_table_split.npy"
                np.save(map_table_name, self.map_table_split)
                print(map_table_name + " saved")
                iter_count += 1
            #     这里可以增加一个画图
            print(max_traffic_iter)
            print(average_traffic_iter)
            print(min_traffic_iter)
            np.save(self.map_path + "output_max.npy", max_traffic_iter)
            np.save(self.map_path + "output_average.npy", average_traffic_iter)
            np.save(self.map_path + "output_min.npy", min_traffic_iter)
        else:
            iter_count = 0
            while iter_count < 70:
                column_idx_to_process = self.allocate_idx_to_calculate()
                self.map_table_split = self.comm.bcast(self.map_table_split, self.master_rank)
                self.map_table_split_old = self.comm.bcast(self.map_table_split_old, self.master_rank)
                self.gpu_changed_idx = self.comm.bcast(self.gpu_changed_idx, self.master_rank)
                self.comm.barrier()
                for gpu_out_idx in column_idx_to_process:
                    traffic_table_base_gpu_out_tmp = np.zeros((self.N,))
                    traffic_table_base_gpu_in_tmp = np.zeros((self.N,))
                    time1 = time.time()
                    # traffic_table_base_gpu_out_tmp, traffic_table_base_gpu_in_tmp = self.calculate_2_dim_output_traffic_for_map_out_in(
                    #     gpu_out_idx)

                    traffic_table_base_gpu_out_tmp, traffic_table_base_gpu_in_tmp = self.calculate_2_dim_output_traffic_for_map_out_in_optimize_input_calculations(
                        gpu_out_idx)

                    time2 = time.time()
                    self.comm.send(traffic_table_base_gpu_out_tmp, dest=self.master_rank)
                    # self.comm.send(traffic_base_gpu_for_a_column_in, dest=self.master_rank)
                    if gpu_out_idx % 1000 == 0:
                        print('Col %d sent: %.4fs consumed.' % (gpu_out_idx, time2 - time1))
                iter_count += 1


if __name__ == "__main__":
    g = GenerateMyMapParallel()
    # g.test_map_sqlite_v2()
    # g.compute_2_dim_traffic_with_pop_traffic()
    g.generate_map_by_output_or_input()
    # g.generate_map_by_output()
