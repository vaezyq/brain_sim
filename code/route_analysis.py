"""
本类用于分析路由表性能
1. 验证路由表准确性，计算不同转发次数的频数
2. 计算以dcu为单位和以节点为单位的连接数
3. 计算各级路由流量
"""
import os.path

import numpy as np
import time
import matplotlib.pyplot as plt
import pickle
from generate_route import GenerateRoute


class RouteAnalysis(GenerateRoute):
    def __init__(self):
        super().__init__()

        self.route_table = self.read_route_npy(self.route_npy_save_path)
        self.traffic_table_base_dcu_path = self.traffic_table_root + 'traffic_table_base_dcu_' + self.map_version + '.npy'
        assert os.path.exists(self.traffic_table_base_dcu_path)
        self.traffic_base_dcu = np.load(self.traffic_table_base_dcu_path)

    def show_info(self):
        print('\n############################')
        print('*******Route Analysis*******')
        print('Route version:     ' + str(self.route_version))
        print('N of dcus:         ' + str(self.N))
        print('N dcus per group:  ' + str(self.n_gpu_per_group))
        print('############################\n')

    # 验证路由表的正确性，并求出一个N*N的矩阵，第i行的第j个元素代表有i号卡发到j需要转发的次数
    def confirm_route_table(self):
        print('Begin to confirm route table...')
        start_time = time.time()

        # 计算src要经过几次转发可以到达dst，同时判断dcu之间是否全部连通
        step_length = np.zeros((self.N, self.N), dtype=np.int)
        for src in range(self.N):
            for dst in range(self.N):
                # 判断是否存在通路，如果不存在通路，会陷入死循环
                temp_src = src
                while self.route_table[temp_src][dst] != temp_src:
                    temp_src = self.route_table[temp_src][dst]
                    step_length[src][dst] += 1
                    assert step_length[src][dst] < 10
            if src % 1000 == 0:
                print('%d / %d' % (src, self.N))
            self.show_progress(src, self.N, start_time)

        print()
        for i in range(np.max(step_length) + 1):
            print('转发次数为%d的频数： %d' % (i, int(np.sum(step_length == i))))
        print('频数总和：%d' % (self.N ** 2))
        print(str(self.route_version) + ' confirmed.')

    # 计算每个dcu的连接数
    def cal_link_in_route_table_base_dcu(self):
        cnt = np.zeros(self.N)
        for i in range(self.N):
            for j in range(self.N):
                if self.route_table[i][j] == i:
                    cnt[i] += 1

        dcu_no = list()
        for i in range(self.N):
            dcu_no.append(i)

        fig_name = self.route_version + '_link_num_base_dcu.png'
        plt.figure(figsize=(10, 6))
        plt.title('conn_number：max=%d, min=%d, average=%.f' % (np.max(cnt) - 1, np.min(cnt) - 1, np.average(cnt) - 1))
        plt.xlabel('dcu')
        plt.ylabel('conn_number')
        plt.plot(cnt - 1, color='blue')
        plt.ylim(60, 120)
        # plt.scatter(dcu_no, cnt-1, color='blue', s=5)
        plt.savefig(self.route_path + fig_name, dpi=200)
        print('Figure ' + fig_name + ' saved.')
        plt.show()
        print('hey')

    # 计算每个节点的连接数
    def cal_link_in_route_table_base_node(self):
        # 计算以节点为单位的矩阵是否有连接
        link_table_base_node = list()
        for i in range(self.number_of_nodes):
            link_table_base_node.append([0] * self.number_of_nodes)
        time1 = time.time()
        for i in range(self.number_of_nodes):
            for j in range(self.number_of_nodes):
                for p in range(4):
                    row = 4 * i + p
                    for q in range(4):
                        col = 4 * j + q
                        if self.route_table[row][col] == row:
                            link_table_base_node[i][j] += 1
            # self.show_progress(i, self.number_of_nodes, time1)

        cnt = np.zeros(self.number_of_nodes)
        for i in range(self.number_of_nodes):
            cnt[i] = self.number_of_nodes - link_table_base_node[i].count(0)

        # 生成0到self.N - 1的序列
        dcu_no = list()
        for i in range(self.number_of_nodes):
            dcu_no.append(i)

        fig_name = self.route_version + '_link_num_base_node_scatter.png'
        plt.figure(figsize=(10, 6))
        plt.title('max=%d, min=%d, average=%.f' % (np.max(cnt) - 1, np.min(cnt) - 1, np.average(cnt) - 1))
        plt.xlabel('node')
        plt.ylabel('conn_number')
        plt.ylim(0, 70)
        plt.scatter(dcu_no, cnt - 1, color='blue', s=7)
        plt.savefig(self.route_path + fig_name, dpi=200)
        print('\nFigure ' + fig_name + ' saved.')

        plt.clf()
        fig_name = self.route_version + '_link_num_base_node_line.png'
        plt.figure(figsize=(10, 6))
        plt.title('max=%d, min=%d, average=%.f' % (np.max(cnt) - 1, np.min(cnt) - 1, np.average(cnt) - 1))
        plt.xlabel('node')
        plt.ylabel('conn_number')
        plt.ylim(0, 70)
        plt.plot(cnt - 1, color='blue')
        plt.savefig(self.route_path + fig_name, dpi=200)
        print('Figure ' + fig_name + ' saved.')

        return

    # 计算1级路由流量
    def cal_level1_flow(self):
        level1_table = list()
        for i in range(self.N):
            level1_table.append([])

        for src in range(self.N):
            for dst in range(self.N):
                temp = self.route_table[src][dst]
                if src // self.n_gpu_per_group == dst // self.n_gpu_per_group:
                    level1_table[temp].append([src, dst])

        time1 = time.time()
        print('Begin calculating level1 route flow...')
        level1_flow = [0] * self.N
        for i in range(self.N):
            for pair in level1_table[i]:
                src, dst = pair[0], pair[1]
                for voxel_out in self.map_table_without_invalid_idx[src]:
                    for voxel_in in self.map_table_without_invalid_idx[dst]:
                        level1_flow[i] += self.conn[voxel_out][voxel_in] * self.size[voxel_out]

            self.show_progress(i, self.N, time1)

        time2 = time.time()
        print('\nCalculation of level1 route flow done: %.2fs consumed.' % (time2 - time1))
        np.save(self.route_path + 'level1_flow.npy', level1_flow)

    # 计算2级路由的流量
    def cal_level2_flow(self):
        level2_table = list()
        for i in range(self.N):
            level2_table.append([])

        for dst in range(self.N):
            for src in range(self.N):
                temp = self.route_table[src][dst]
                if src // self.n_gpu_per_group != dst // self.n_gpu_per_group:
                    level2_table[temp].append([src, dst])

        time1 = time.time()
        print('Begin calculating level2 route flow...')
        cnt = 0
        level2_flow = [0] * self.N
        for i in range(self.N):
            for pair in level2_table[i]:
                src, dst = pair[0], pair[1]
                for voxel_out in self.map_table[src]:
                    for voxel_in in self.map_table[dst]:
                        level2_flow[i] += self.conn[voxel_out][voxel_in] * self.size[voxel_out]
                        cnt += 1

            self.show_progress(i, self.N, time1)

        time2 = time.time()
        print('\nCalculation of level2 route flow done: %.2fs consumed.' % (time2 - time1))
        np.save(self.route_path + 'level2_flow.npy', level2_flow)
        print('hey')

    def cal_level2_out_traffic_new(self):
        traffic_table_base_dcu = np.load('../tables/traffic_table/traffic_table_base_dcu_map_10000_v1_cortical_v1.npy')
        level2_out_traffic = np.zeros(self.N)

        start_time = time.time()

        for src in range(self.N):
            for dst in range(self.N):
                if self.route_table[src][dst] != src:
                    bridge = self.route_table[src][dst]
                    level2_out_traffic[bridge] += traffic_table_base_dcu[dst][src]

            self.show_progress(src, self.N, start_time)

        plt.title('max=%d, average=%d, max/average=%.4f' % (np.max(level2_out_traffic), np.average(level2_out_traffic),
                                                            np.max(level2_out_traffic) / np.average(
                                                                level2_out_traffic)))
        # plt.ylim(0.7e8, 3e8)
        plt.plot(level2_out_traffic)
        plt.show()

        # np.save('level2_flow.npy', level2_out_traffic)

        # np.save('./IPM/PGA_data_traffic.npy', level2_out_traffic)
        # print('./IPM/PGA_data_traffic.npy saved.')

        print('hey')

    def show_two_level_traffic(self, level1_traffic, level2_traffic, name):
        print('level1:', np.min(level1_traffic), np.max(level1_traffic), np.average(level1_traffic))
        print('level2:', np.min(level2_traffic), np.max(level2_traffic), np.average(level2_traffic))

        plt.figure(figsize=(12, 6), dpi=200)
        plt.title('level1 traffic: max = %.f, average = %.f, max/average = %.2f' %
                  (np.max(level1_traffic), np.average(level1_traffic),
                   np.max(level1_traffic) / np.average(level1_traffic)))
        plt.ylim(500000, 260000000)
        plt.plot(level1_traffic, color='red', alpha=0.5, linewidth=0.1)
        plt.savefig(self.route_path + name + '_level1_traffic.png')
        plt.show()
        plt.close()

        plt.figure(figsize=(12, 6), dpi=200)
        plt.title('level2 traffic: max = %.f, average = %.f, max/average = %.2f' %
                  (np.max(level2_traffic), np.average(level2_traffic),
                   np.max(level2_traffic) / np.average(level2_traffic)))
        plt.ylim(500000, 260000000)
        plt.plot(level2_traffic, color='blue', alpha=0.7, linewidth=0.1)
        plt.savefig(self.route_path + name + '_level2_traffic.png')
        plt.show()
        plt.close()

    def cal_two_level_traffic_in(self):
        traffic_table_base_dcu = np.load('../tables/map_table/traffic_table_base_dcu_17280.npy')
        route_table = self.route_table

        level1_traffic_in = np.zeros(self.N)
        level2_traffic_in = np.zeros(self.N)

        time1 = time.time()
        for i in range(self.N):
            # 先统计各个节点负责的dcu有哪些
            masters_and_slaves = dict()
            for j in range(self.N):
                key = route_table[i][j]
                if key not in masters_and_slaves:
                    masters_and_slaves[key] = list()
                masters_and_slaves[key].append(j)

            for j in range(self.N):
                # if j in masters_and_slaves[i]:
                #     level1_traffic_in[j] += traffic_table_base_dcu[j][i]
                # else:
                #     level2_traffic_in[j] += traffic_table_base_dcu[j][i]

                level1_traffic_in[j] += traffic_table_base_dcu[j][i]
                if j not in masters_and_slaves[i]:
                    level2_traffic_in[j] += traffic_table_base_dcu[j][i]
            if i % 1000 == 0:
                print('%d / %d' % (i, self.N))
        time2 = time.time()
        print('%.2f seconds consumed.' % (time2 - time1))
        np.save(self.route_path + 'level1_traffic_in.npy', level1_traffic_in)
        np.save(self.route_path + 'level2_traffic_in.npy', level2_traffic_in)
        self.show_two_level_traffic(level1_traffic_in, level2_traffic_in, name='in')

    def cal_two_level_traffic_out(self):
        traffic_table_base_dcu = np.load('../tables/map_table/traffic_table_base_dcu_17280.npy')
        route_table = self.route_table

        level1_traffic_out = np.zeros(self.N)
        level2_traffic_out = np.zeros(self.N)

        time1 = time.time()
        for i in range(self.N):
            # 先统计各个节点负责的dcu有哪些
            masters_and_slaves = dict()
            for j in range(self.N):
                key = route_table[i][j]
                if key not in masters_and_slaves:
                    masters_and_slaves[key] = list()
                masters_and_slaves[key].append(j)

            for j in range(self.N):
                key = route_table[i][j]
                # if j in masters_and_slaves[i]:
                #     level1_traffic_out[i] += traffic_table_base_dcu[j][i]
                # else:
                #     level2_traffic_out[key] += traffic_table_base_dcu[j][i]
                level1_traffic_out[i] += traffic_table_base_dcu[j][i]
                if j not in masters_and_slaves[i]:
                    level2_traffic_out[key] += traffic_table_base_dcu[j][i]

            if i % 1000 == 0:
                print('%d / %d' % (i, self.N))
        time2 = time.time()
        print('%.2f seconds consumed.' % (time2 - time1))
        np.save(self.route_path + 'level1_traffic_out.npy', level1_traffic_out)
        np.save(self.route_path + 'level2_traffic_out.npy', level2_traffic_out)
        self.show_two_level_traffic(level1_traffic_out, level2_traffic_out, name='out')

    def show_comparison_chart(self):
        path1 = self.route_path + 'level1_flow.npy'
        path2 = self.route_path + 'level2_flow.npy'

        level1_flow, level2_flow = np.load(path1), np.load(path2)
        flow_sum = level1_flow + level2_flow

        # plt.title('两个版本的2级路由流量对比')
        # plt.title('1级2级路由流量对比')
        plt.figure(figsize=(10, 6))
        plt.title('max=%.6f, min=%.6f, average=%.6f' % (np.max(flow_sum), np.min(flow_sum), np.average(flow_sum)))
        # plt.title('max=%.6f, min=%.6f, average=%.6f' % (np.max(level1_flow), np.min(level1_flow),
        #                                                 np.average(level1_flow)))
        # plt.title('max=%.6f, min=%.6f, average=%.6f' % (np.max(level2_flow), np.min(level2_flow),
        #                                                 np.average(level2_flow)))

        plt.xlabel('dcu')
        plt.ylabel('traffic')
        # plt.yticks([0, 0.0002, 0.0004, 0.0006, 0.0008, 0.001])
        # plt.ylim(0, 0.0008)
        plt.plot(level1_flow, color='red', alpha=0.2, label='1级', linewidth=1)
        plt.plot(level2_flow, color='blue', alpha=0.2, label='2级', linewidth=1)
        plt.plot(flow_sum, color='green', alpha=1, label='sum', linewidth=1.5)
        plt.legend(fontsize=17)
        plt.savefig(self.route_path + 'flow_comparison.png', dpi=200)
        plt.show()

        print('hey')

    @staticmethod
    def compare_flow_of_different_version():
        path1 = ''
        path2 = ''
        flow_version1, flow_version2 = np.loadtxt(path1), np.loadtxt(path2)

        plt.title('max_old=%.6f, max_new=%.6f' % (np.max(flow_version1), np.max(flow_version2)))
        plt.xlabel('dcu')
        plt.ylabel('traffic')
        plt.ylim(0, 0.001)
        plt.plot(flow_version1, color='green', alpha=0.3, linestyle='--', label='old', linewidth=2)
        plt.plot(flow_version2, color='green', label='new', linewidth=1.5)
        plt.legend(fontsize=17)
        plt.show()

        print('hey')

    # 计算经过路由后，dcu之间的真实连接数，需要大约22分钟
    def cal_genuine_link_number_base_dense_route_table(self, number_of_group, dcu_per_group):
        binary_connection_table_base_dcu = np.load(self.traffic_table_base_dcu_path)
        binary_connection_table_base_dcu = np.array(binary_connection_table_base_dcu, dtype=bool)
        # route_table = self.generate_route_default(self.N, number_of_group=number_of_group, dcu_per_group=dcu_per_group)
        route_table = np.load(self.route_path + 'route.npy')

        inside_group_link_num = np.zeros(self.N, dtype=int)
        between_group_link_num = np.zeros(self.N, dtype=int)

        time1 = time.time()
        for i in range(self.N):
            # 先统计各个节点负责的dcu有哪些
            masters_and_slaves = dict()
            for j in range(self.N):
                temp = route_table[i][j]
                if temp not in masters_and_slaves:
                    masters_and_slaves[temp] = list()
                masters_and_slaves[temp].append(j)

            for j in range(self.N):
                if j not in masters_and_slaves and j in masters_and_slaves[i]:  # 组间
                    if binary_connection_table_base_dcu[j][i] == 1:
                        inside_group_link_num[i] += 1
                elif j in masters_and_slaves:  # 组内
                    for idx in masters_and_slaves[j]:
                        if binary_connection_table_base_dcu[idx][i] == 1:
                            between_group_link_num[i] += 1
                            break
            if i % 1000 == 0:
                print(i)
        time2 = time.time()

        link_num = inside_group_link_num + between_group_link_num
        np.save('inside_group_%d_%d.npy' % (number_of_group, dcu_per_group), inside_group_link_num)
        np.save('between_group_%d_%d.npy' % (number_of_group, dcu_per_group), between_group_link_num)
        np.save('link_num_%d_%d.npy' % (number_of_group, dcu_per_group), link_num)
        print('########### %d * %d ###########' % (number_of_group, dcu_per_group))
        print('inside group:', np.max(inside_group_link_num), np.min(inside_group_link_num),
              np.average(inside_group_link_num))
        print('between group:', np.max(between_group_link_num), np.min(between_group_link_num),
              np.average(between_group_link_num))
        print('sum:', np.max(link_num), np.min(link_num), np.average(link_num))
        print('%.2f seconds consumed.' % (time2 - time1))

    def check_connection(self, src, dst):
        result = False
        for cortical_src in self.map_table_without_invalid_idx[str(src)]:
            for cortical_dst in self.map_table_without_invalid_idx[str(dst)]:
                if self.conn[cortical_dst][cortical_src] != 0:
                    result = True
                    break

        print(result)

    '''
    测试在不同分组方案下，真实连接数的情况
    '''
    def temp(self):
        dcu_per_group_list = [120, 160]
        for dcu_per_group in dcu_per_group_list:
            self.cal_genuine_link_number_base_dense_route_table(int(self.N / dcu_per_group), dcu_per_group)


if __name__ == "__main__":
    Job = RouteAnalysis()
    Job.show_info()
    Job.confirm_route_table()
    Job.cal_link_in_route_table_base_dcu()
