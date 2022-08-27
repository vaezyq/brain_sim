"""
这个类中包含用于计算过程与读取文件的辅助函数
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from data import Data


class Functions(Data):
    def __init__(self):
        super().__init__()

        self.max_range = [0, 0]
        self.max_times = [0, 0]

        self.traffic_table_root = self.root_path + 'tables/traffic_table/'
        self.out_traffic, self.in_traffic = np.array([]), np.array([])
        self.voxels_per_dcu = self.generate_voxels_per_dcu()

        self.plt_show_minus()
        self.counter = 0  # 用于显示进度条

    @staticmethod
    def plt_show_chinese():
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文

    @staticmethod
    def plt_show_minus():
        plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

    # 展示计算进度条、预计剩余时间
    def show_progress(self, num, base, start_time, width=37):
        if self.if_show_progress:
            percent = 100 * num / (base - 1)
            left = width * int(percent) // 100
            mid = 1 if left != width else 0
            right = width - left - 1 if left != width else 0

            marks = ['-', '\\', '|', '/']
            mark = marks[self.counter % 4]
            self.counter += 1

            now_time = time.time()
            time_left = int((now_time - start_time) * (100 - percent) / (percent + 0.00001))

            print('\r[', '#' * left, mark * mid, ' ' * right, ']', f' {percent:.1f}%  ',
                  f'{(time_left // 60):d}m' + f'{(time_left % 60):d}s', ' left...',
                  sep='', end='', flush=True)
        else:
            return

    # 计算初始情况下每个dcu应该模拟多少个体素，计算过程相当于解二元一次方程
    def generate_voxels_per_dcu(self):
        average_dcu_num = self.n // self.N
        y = self.n - average_dcu_num * self.N
        x = self.N - y
        dic = {average_dcu_num: x, (average_dcu_num + 1): y}
        return dic

    # 以概率*size 的方式计算流量
    def cal_traffic_old(self, loc, traffic_direction):
        traffic = 0
        assert (traffic_direction == 'out' or traffic_direction == 'in')

        if traffic_direction == 'out':
            traffic = self.conn[loc, :] * self.size
        if traffic_direction == 'in':  # 这里有问题
            traffic = self.conn[:, loc] * self.size

        return traffic

    # 以采样的方式计算流量
    def cal_traffic_with_sampling(self, loc):
        import torch
        torch.cuda.empty_cache()
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        if torch.cuda.is_available():
            print("\n%.fMB VRAM allocated." % (torch.cuda.memory_allocated(device=device) / 10000000))

        start_time = time.time()

        '''
        conn_number_estimate是一个(self.N, 1)维的array
        conn_number_estimate[i]代表第i号voxel发往第loc号voxel的消息量
        '''
        conn_number_estimate = self.neuron_number * self.size * self.degree * self.conn[:, loc]

        # 计算采样范围与采样次数
        sample_range = int(self.neuron_number * self.size[loc])
        sample_times = int(np.sum(conn_number_estimate))
        print("range: %d, sample times: %d" % (sample_range, sample_times))

        if sample_range > self.max_range[0]:
            self.max_range = [sample_range, sample_times]

        if sample_times > self.max_times[1]:
            self.max_times = [sample_range, sample_times]

        print('max_range:', self.max_range)
        print('max_times:', self.max_times)

        data = list()
        n_slice = 100
        for i in range(n_slice):
            temp = torch.unique(torch.randint(0, int(sample_range), (int(sample_times / n_slice),), device=device))
            data.append(temp.clone())
            torch.cuda.empty_cache()
        # print("%.fMB VRAM allocated." % (torch.cuda.memory_allocated(device=device) / 1000000))
        del temp

        new_data = torch.cat(data)
        new_data = torch.unique(new_data)
        traffic = torch.unique(new_data).numel()

        end_time = time.time()
        print('loc %d: %.4f' % (loc, end_time - start_time))

        # return conn_number_estimate, traffic
        return traffic

    # 计算每个体素以概率*size的方式计算的流量
    def cal_probability_mul_size(self):
        out_traffic = np.array([])
        in_traffic = np.array([])

        # start_time = time.time()
        for i in range(self.n):
            out_traffic = np.append(out_traffic, self.cal_traffic_with_sampling(i))
            # in_traffic = np.append(in_traffic, self.cal_traffic_old(i, traffic_direction='in'))
            # self.show_progress(i, self.n, start_time)

        return out_traffic, in_traffic

    # 计算每个体素以采样方式计算的流量
    def cal_traffic_base_voxel(self):
        out_traffic = np.zeros(self.n)
        in_traffic = np.array(self.n)
        # out_traffic_voxel_to_voxel = np.zeros((self.n, self.n))

        start_time = time.time()
        for i in range(self.n):
            # out_traffic_voxel_to_voxel[i], out_traffic[i] = self.cal_traffic_with_sampling(i)
            out_traffic[i] = self.cal_traffic_with_sampling(i)
        end_time = time.time()
        print('traffic base voxel calculation completed. %.2fs consumed.' % (end_time - start_time))

        name1 = "out_traffic_per_voxel_" + str(self.neuron_number) + ".npy"
        np.save(self.traffic_table_root + name1, out_traffic)
        print('%s saved.' % name1)

        # 保存每个元素到每个元素的流量
        # out_traffic_voxel_to_voxel = out_traffic_voxel_to_voxel.transpose()  # 转置
        # name2 = "traffic_table_base_voxel_" + str(self.neuron_number) + ".npy"
        # np.save(self.traffic_table_path + name2, out_traffic_voxel_to_voxel)
        # print('%s saved.' % name2)

        return out_traffic, in_traffic

    # 计算分组后，每张dcu上的所有体素的流量之和
    def sum_traffic_per_dcu(self, map_table):
        sum_out_under_dcu = list()
        sum_in_under_dcu = list()

        for i in range(self.N):
            sum_traffic = 0
            for idx in map_table[i]:
                sum_traffic += self.out_traffic[idx]
            sum_out_under_dcu.append(sum_traffic)
            # sum_in_under_dcu.append(np.sum(self.in_traffic[np.ix_(map_table[i])]))

        return np.array(sum_out_under_dcu), np.array(sum_in_under_dcu)

    # 读入.mat格式的连接概率数据
    @staticmethod
    def read_mat():
        from scipy import io
        folder_path = '../tables/conn_table/voxel_v2/'
        file_name = 'DTI_voxel_network_mat_1115_dx.mat'
        data = io.loadmat(folder_path + file_name)
        voxel_size = data['voxel_size'].reshape(-1)
        conn = data['dti']

        np.save(folder_path + 'conn.npy', conn)
        print(folder_path + 'conn.npy saved.')

        np.save(folder_path + 'size.npy', voxel_size)
        print(folder_path + 'size.npy saved.')

    # 读入.pkl格式的映射表，以list的形式返回
    def read_map_pkl(self, path):
        import pickle
        f = open(path, 'rb')
        map_table = pickle.load(f)

        map_list = list()
        for i in range(self.N):
            map_list.append(map_table[str(i)])

        # print('Map table(.pkl) loaded.')

        return map_list

    # 读入.pkl格式的映射表，list形式返回（用于初始未分割的映射表）
    def read_map_dict_pkl(self, path):
        import pickle
        import numpy as np
        f = open(path, 'rb')
        map_table = pickle.load(f)

        map_list = list()
        for i in range(self.N):
            b = np.ones((len(map_table[str(i)])))
            map_list.append(dict(zip(map_table[str(i)], b)))

        # print('Map table(.pkl) loaded.')

        return map_list

    # 将dcu-体素映射表保存为pkl文件
    def save_map_pkl(self, map_table, path):
        # 将映射表转化为字典的格式
        map_pkl = dict()
        for i in range(self.N):
            map_pkl[str(i)] = map_table[i]

        import pickle
        with open(path, 'wb') as f:
            pickle.dump(map_pkl, f)

        print('%s saved.' % path)

    # 读入txt格式的路由表，以np.array()的形式返回
    @staticmethod
    def read_route_npy(path):
        data = np.load(path, allow_pickle=True)
        # print('Route table(.npy) loaded.')
        return data

    # 读入json格式的路由表，以np.array()的形式返回
    def read_route_json(self):
        pass

    # 将路由表保存为json文件
    def save_route_json(self, route_table, route_saving_path):
        """
        N = 10000: 20 seconds + 3 minutes needed.
        """
        route_table = route_table.tolist()

        # 把路由表中自己发给自己的部分去掉
        new_route = list()
        for i in range(self.N):
            del route_table[i][i]
            new_route.append(route_table[i])

        # 将路由表转化为字典的格式，以便保存为json文件
        start_time = time.time()
        route_dic = dict()
        for i in range(self.N):
            route_dic[str(i)] = dict()

            # 生成src
            route_dic[str(i)]['src'] = new_route[i]

            # 生成dst
            dst = list()
            for j in range(self.N):
                if j != i:
                    dst.append(j)
            route_dic[str(i)]['dst'] = dst
            # self.show_progress(i, self.N, start_time)

        # 将路由表保存为json文件
        import json
        route_json = json.dumps(route_dic, indent=2, sort_keys=False)
        with open(route_saving_path, 'w') as json_file:
            json_file.write(route_json)

        end_time = time.time()
        print('\n%s saved. %2.fs consumed.' % (route_saving_path, (end_time - start_time)))


if __name__ == "__main__":
    Job = Functions()
    # route_table = np.load('../tables/route_table/route_v1_map_10000_v3_cortical_v2/route.npy')
    # route_saving_path = '../tables/route_table/route_v1_map_10000_v3_cortical_v2/route_dense.json'
    # Job.save_route_json(route_table, route_saving_path)
