"""
路由表的生成过程
"""

import numpy as np
import time
from map_analysis import MapAnalysis
import os


class GenerateRoute(MapAnalysis):
    def __init__(self):
        super().__init__()

        self.method_dict = {'default': self.generate_route_default_2dim,
                            '2': self.generate_route_level2,
                            '5': self.generate_route_level5,
                            'new': self.generate_route_average,
                            'less_link': self.generate_route_less_link,
                            'n_gpu_per_group': self.generate_route_n_gpu_per_group,
                            'base_node': self.generate_route_base_node,
                            'unpack_inside_group': self.generate_route_unpack_inside_group,
                            'unpack_inside_group_sampling': self.generate_route_unpack_inside_group_sampling,
                            '17280_v1': self.generate_route_17280_v1}

        # print('Initialize complete.')

    # 把形式为矩阵的映射表转化为<group，dcu号，包含的体素编号>的三元组格式
    def generate_tuples(self):
        tuples = list()
        for dcu_idx in range(self.N):
            single_dcu = dict()
            single_dcu['idx'] = dcu_idx  # dcu的绝对编号
            single_dcu['group_idx'] = dcu_idx // self.n_gpu_per_group  # 单张dcu的group号
            single_dcu['node_idx'] = (dcu_idx % self.n_gpu_per_group) // 4  # node的相对编号，范围[0, 19]
            single_dcu['dcu_idx'] = dcu_idx % 4  # dcu的相对编号，范围[0, 3]
            # single_dcu['voxels'] = self.map_table[dcu_idx]
            tuples.append(single_dcu)

        return tuples

    # 根据交换机编号，节点编号和节点内dcu编号计算路由表中的dcu编号
    def cal_gpu_number(self, tetrad):
        dcu_no = tetrad['group_idx'] * self.n_gpu_per_group + tetrad['node_idx'] * 4 + tetrad['dcu_idx']
        return int(dcu_no)

    # 把稠密格式的路由表转化为稀疏格式
    def to_sparse_route(self):
        pass

    # 把稀疏格式的路由表转化为稠密格式
    def to_dense_route(self):
        pass

    # 生成路由表的主要逻辑，具体生成方法从self.method_dict中选择
    def generate_route(self, method=None):
        start_time = time.time()
        print('Begin to generate route table...')

        # 转化为四元组格式
        tuples = self.generate_tuples()

        # 生成路由表的主要逻辑
        assert self.method_dict[method] is not None
        func = self.method_dict.get(method)
        route_table = func(tuples)

        end_time = time.time()
        print('\nRoute table generated. %.2fs consumed.' % (end_time - start_time))

        np.save(self.route_npy_save_path, route_table)
        self.save_route_json(route_table, self.route_dense_json_save_path)

    # 生成2层路由
    def generate_route_level2(self, tuples):
        # 生成空路由表
        route_table = list()
        for i in range(self.N):
            route_table.append([0] * self.N)

        for dcu_out in tuples:
            for dcu_in in tuples:  # 对于任意两个dcu生成路由表
                dcu_out_idx, dcu_in_idx = self.cal_gpu_number(dcu_out), self.cal_gpu_number(dcu_in)

                if dcu_out['switch_no'] == dcu_in['switch_no']:  # 如果两张卡在同一交换机下
                    route_table[dcu_out_idx][dcu_in_idx] = dcu_out_idx
                else:  # 两张卡不在同一交换机下
                    # 判断是否为本节点负责的，如果是，dcu_out_idx
                    if dcu_in['switch_no'] % self.node_per_switch == dcu_out['node_no'] and dcu_out['dcu_no'] == 0:
                        route_table[dcu_out_idx][dcu_in_idx] = dcu_out_idx
                    else:
                        # src = dcu_out['switch_no'] * self.node_per_switch * self.dcu_per_node + \
                        #       (dcu_in['switch_no'] % self.node_per_switch) * self.dcu_per_node + \
                        #       np.random.randint(0, 4)
                        src = dcu_out['switch_no'] * self.node_per_switch * self.gpu_per_node + \
                              (dcu_in['switch_no'] % self.node_per_switch) * self.gpu_per_node
                        route_table[dcu_out_idx][dcu_in_idx] = src
            # print(dcu_out['switch_no'])  # 显示进度
        return route_table

    # 生成5层路由
    def generate_route_level5(self, tuples):
        # 生成空路由表
        route_table = list()
        for i in range(self.N):
            route_table.append([0] * self.N)

        for dcu_out in tuples:
            for dcu_in in tuples:  # 对于任意两个dcu生成路由表
                dcu_out_idx, dcu_in_idx = self.cal_gpu_number(dcu_out), self.cal_gpu_number(dcu_in)
                if dcu_out['group_idx'] == dcu_in['group_idx']:  # 如果两张卡在同一group下
                    if dcu_out['node_idx'] == dcu_in['node_idx']:  # 如果在同一节点下，填自己的编号
                        route_table[dcu_out_idx][dcu_in_idx] = dcu_out_idx
                    elif dcu_out['dcu_idx'] == dcu_in['dcu_idx']:  # 如果dcu编号相同，填自己的编号
                        route_table[dcu_out_idx][dcu_in_idx] = dcu_out_idx
                    else:  # 如果不在同一节点下，通过对应节点转发 ???????????????????
                        route_table[dcu_out_idx][dcu_in_idx] = dcu_in['group_idx'] * self.n_gpu_per_group + \
                                                               dcu_in['node_idx'] * 4 + dcu_out['dcu_idx']
                else:  # 两张卡不在同一交换机下
                    if dcu_in['group_idx'] % self.n_node_per_group == dcu_out['node_idx']:  # 如果是本节点负责的交换机
                        if dcu_in['node_idx'] == 0 and dcu_in['dcu_idx'] == 0 and dcu_out['dcu_idx'] == 0:  # for j in range(iter_times): # 如果发送方是0号dcu，接收方是0号节点0号dcu，直接发送
                            route_table[dcu_out_idx][dcu_in_idx] = dcu_out_idx
                        elif dcu_out['dcu_idx'] == 0:  # 如果发送方是0号dcu，接收方不是0号节点0号dcu，则发给指定交换机的0号节点的0号dcu
                            route_table[dcu_out_idx][dcu_in_idx] = dcu_in['group_idx'] * self.n_gpu_per_group
                        else:  # 如果发送方不是0号dcu，则发给同一节点的0号dcu
                            route_table[dcu_out_idx][dcu_in_idx] = dcu_out['group_idx'] * self.n_gpu_per_group + \
                                                                   dcu_out['node_idx'] * 4
                    else:  # 如果不是本节点负责的交换机，发给同一group内的指定节点的0号dcu
                        if dcu_out['dcu_idx'] == 0:  # 如果发送方是0号dcu，直接发送给同一group下指定节点的0号dcu
                            route_table[dcu_out_idx][dcu_in_idx] = dcu_out['group_idx'] * self.n_gpu_per_group \
                                                                   + (dcu_in['group_idx'] % self.n_node_per_group) * 4
                        else:  # 如果发送方不是0号dcu，首先转发给0号dcu
                            route_table[dcu_out_idx][dcu_in_idx] = dcu_out['group_idx'] * self.n_gpu_per_group \
                                                                   + dcu_out['node_idx'] * 4
            # print(dcu_out['group_idx'])

        return route_table

    # 按二级路由均分原则进行路由转发
    def generate_route_average(self, tuples):

        forwarding_table = self.cal()

        # 生成空路由表
        route_table = list()
        for i in range(self.N):
            route_table.append([0] * self.N)

        time1 = time.time()
        for dcu_out in tuples:
            single_forwarding_table = forwarding_table[dcu_out['switch_no']]
            for dcu_in in tuples:  # 对于任意两个dcu生成路由表
                dcus_to_forward = single_forwarding_table[dcu_in['switch_no']]
                dcu_out_idx, dcu_in_idx = self.cal_gpu_number(dcu_out), self.cal_gpu_number(dcu_in)

                if dcu_out['switch_no'] == dcu_in['switch_no']:  # 如果两张卡在同一交换机下
                    route_table[dcu_out_idx][dcu_in_idx] = int(dcu_out_idx)
                else:  # 两张卡不在同一交换机下
                    # 判断是否为本dcu负责的，如果是，dcu_out_idx,否则从负责接受消息的交换机的dcu中随机选一个进行转发
                    if dcu_out_idx in dcus_to_forward:
                        route_table[dcu_out_idx][dcu_in_idx] = int(dcu_out_idx)
                    else:
                        src = np.random.choice(dcus_to_forward)
                        route_table[dcu_out_idx][dcu_in_idx] = int(src)
                # self.show_progress(dcu_out_idx, self.N, time1)

        return route_table

    # 相比于上个方法，减少了连接数
    def generate_route_less_link(self, tuples, iter_times=1000):
        forwarding_table = self.cal_forwarding_table()

        # 生成每个交换机向外发送消息时，每个dcu负责哪些dst的转发
        random_name = list()
        for i in range(self.number_of_switches):
            random_name.append([])
            for j in range(self.gpu_per_switch):
                random_name[i].append([])

        time1 = time.time()
        for switch_out in range(self.number_of_switches):
            for switch_in in range(self.number_of_switches):
                n = len(forwarding_table[switch_out][switch_in])  # 负责该交换机的dcu的个数
                responsible_dcus = [self.gpu_per_switch // n] * n if n != 0 else list()
                loc = 0
                if n != 0:
                    for i in range(80 % n):
                        responsible_dcus[loc] += 1
                        loc += 1

                dcu_no = switch_in*80

                for i in range(n):
                    dcu_idx = forwarding_table[switch_out][switch_in][i] % 80
                    for j in range(responsible_dcus[i]):
                        random_name[switch_out][dcu_idx].append(dcu_no)
                        dcu_no += 1
        time2 = time.time()
        print('done. %.2fs consumed.' % (time2-time1))

        num_of_forwarding_dcu = list()
        for i in range(self.number_of_switches):
            num_of_forwarding_dcu.append([])
            for j in range(self.number_of_switches):
                num_of_forwarding_dcu[i].append(len(forwarding_table[i][j]))

        # 根据流量均衡原则重新分配random_name
        for i in range(self.number_of_switches):
            flow_per_dcu = list()

            # 计算flow_per_dcu
            flow_per_dcu = [0] * self.gpu_per_switch
            for j in range(self.gpu_per_switch):
                for dcu_in in random_name[i][j]:
                    flow_per_dcu[j] += self.conn[80 * i + j][dcu_in] * self.size(80 * i + j)

        # 生成空路由表
        route_table = list()
        for i in range(self.N):
            route_table.append([0] * self.N)

        time1 = time.time()
        for dcu_out in tuples:
            out_switch_idx, dcu_out_idx = dcu_out['switch_no'], self.cal_gpu_number(dcu_out)
            single_random_name = random_name[out_switch_idx]

            for dcu_in in tuples:  # 对于任意两个dcu生成路由表
                in_switch_idx, dcu_in_idx = dcu_in['switch_no'], self.cal_gpu_number(dcu_in)

                if out_switch_idx == in_switch_idx:  # 如果在同一交换机下
                    route_table[dcu_out_idx][dcu_in_idx] = int(dcu_out_idx)
                else:  # 两张卡不在同一交换机下
                    # 判断是否为本dcu负责的，如果是，dcu_out_idx,否则用负责接受消息的交换机的dcu进行转发
                    if dcu_in_idx in single_random_name[dcu_out_idx % self.gpu_per_switch]:
                        route_table[dcu_out_idx][dcu_in_idx] = int(dcu_out_idx)
                    else:
                        src = 0
                        for i in range(self.gpu_per_switch):
                            if dcu_in_idx in single_random_name[i]:
                                src = out_switch_idx*80 + i
                        route_table[dcu_out_idx][dcu_in_idx] = int(src)
                # self.show_progress(dcu_out_idx, self.N, time1)

        return route_table

    # 每80个dcu分为一个group
    def generate_route_n_gpu_per_group(self, tuples):
        forwarding_table = self.generate_forwarding_table(iter_times=500)

        # 生成空路由表
        route_table = list()
        for i in range(self.N):
            route_table.append([0] * self.N)

        start_time = time.time()
        for dcu_out in tuples:
            out_group_idx, dcu_out_idx = dcu_out['group_idx'], self.cal_gpu_number(dcu_out)
            single_forwarding_table = forwarding_table[out_group_idx]

            for dcu_in in tuples:  # 对于任意两个dcu生成路由表
                in_group_idx, dcu_in_idx = dcu_in['group_idx'], self.cal_gpu_number(dcu_in)

                if out_group_idx == in_group_idx:  # 如果在同一交换机下
                    route_table[dcu_out_idx][dcu_in_idx] = int(dcu_out_idx)
                else:  # 两张卡不在同一交换机下
                    # 判断是否为本dcu负责的，如果是，dcu_out_idx,否则用负责接受消息的交换机的dcu进行转发
                    if dcu_in_idx in single_forwarding_table[dcu_out_idx % self.n_gpu_per_group]:
                        route_table[dcu_out_idx][dcu_in_idx] = int(dcu_out_idx)
                    else:
                        src = 0
                        for i in range(self.n_gpu_per_group):
                            if dcu_in_idx in single_forwarding_table[i]:
                                src = out_group_idx * self.n_gpu_per_group + i
                                break
                        route_table[dcu_out_idx][dcu_in_idx] = int(src)
            # self.show_progress(dcu_out_idx, self.N, start_time)

        return route_table

    # 以节点为单位生成路由表
    def generate_route_base_node(self, tuples):
        # 此处的转发表以节点为单位
        # forwarding_table[self.number_of_groups][self.n_node_per_group][]
        forwarding_table = self.generate_forwarding_table_base_node()

        # 生成空路由表
        route_table = list()
        for i in range(self.N):
            route_table.append([0] * self.N)

        start_time = time.time()
        for dcu_out in tuples:
            out_group_idx, dcu_out_idx = dcu_out['group_idx'], self.cal_gpu_number(dcu_out)
            out_node_abs_idx = dcu_out['group_idx'] * self.n_node_per_group + dcu_out['node_idx']
            # single_forwarding_table[20][]
            single_forwarding_table = forwarding_table[out_group_idx]

            for dcu_in in tuples:
                in_group_idx, dcu_in_idx = dcu_in['group_idx'], self.cal_gpu_number(dcu_in)
                in_node_abs_idx = dcu_in['group_idx'] * self.n_node_per_group + dcu_in['node_idx']

                if out_group_idx == in_group_idx:  # 如果in、out在同一group下
                    if out_node_abs_idx == in_node_abs_idx:  # 如果out、in在同一node中，直接发送
                        route_table[dcu_out_idx][dcu_in_idx] = dcu_out_idx
                    elif dcu_out['dcu_idx'] != 0:  # 如果out不是0号dcu，发送给相同节点的0号dcu
                        route_table[dcu_out_idx][dcu_in_idx] = dcu_out['group_idx'] * self.n_gpu_per_group\
                                                               + dcu_out['node_idx'] * 4
                    elif dcu_out['dcu_idx'] == 0 and dcu_in['dcu_idx'] == 0:  # 如果out、in都是0号dcu，直接发送
                        route_table[dcu_out_idx][dcu_in_idx] = dcu_out_idx
                    else:  # 如果in不是0号dcu，发送给in对应node的0号dcu（此时out一定是0号dcu）
                        route_table[dcu_out_idx][dcu_in_idx] = dcu_in['group_idx'] * self.n_gpu_per_group\
                                                               + dcu_in['node_idx'] * 4
                else:  # in、out不在同一group下，需要通过转发表判断如何转发
                    if in_node_abs_idx in single_forwarding_table[dcu_out['node_idx']]:  # dcu_in是本node负责转发的
                        if dcu_out['dcu_idx'] == 0 and dcu_in['dcu_idx'] == 0:
                            route_table[dcu_out_idx][dcu_in_idx] = dcu_out_idx
                        elif dcu_out['dcu_idx'] == 0 and dcu_in['dcu_idx'] != 0:
                            route_table[dcu_out_idx][dcu_in_idx] = dcu_in['group_idx'] * self.n_gpu_per_group \
                                                                   + dcu_in['node_idx'] * 4
                        else:
                            route_table[dcu_out_idx][dcu_in_idx] = dcu_out['group_idx'] * self.n_gpu_per_group\
                                                               + dcu_out['node_idx'] * 4
                    else:  # dcu_in不是本node负责转发的，转发给负责的节点
                        src = -1  # src为负责转发的节点的0号dcu的绝对编号
                        for i in range(self.n_node_per_group):
                            if in_node_abs_idx in single_forwarding_table[i]:
                                src = out_group_idx * self.n_gpu_per_group + 4 * i
                                break

                        if dcu_out_idx == src and dcu_in['dcu_idx'] == 0:  # out不负责目标节点且in是目标节点的0号dcu
                            route_table[dcu_out_idx][dcu_in_idx] = dcu_out_idx
                        elif dcu_out_idx == src and dcu_in['dcu_idx'] != 0:  # out不负责目标节点且in不是目标节点的0号dcu
                            route_table[dcu_out_idx][dcu_in_idx] = dcu_in['group_idx'] * self.n_gpu_per_group\
                                                               + dcu_in['node_idx'] * 4
                        else:  # 如果out不是负责目标节点的节点
                            if dcu_out['dcu_idx'] == 0:
                                route_table[dcu_out_idx][dcu_in_idx] = src
                            else:
                                route_table[dcu_out_idx][dcu_in_idx] = dcu_out['group_idx'] * self.n_gpu_per_group\
                                                               + dcu_out['node_idx'] * 4

            # self.show_progress(dcu_out_idx, self.N, start_time)

        return route_table

    # 在group内拆包的方法
    def generate_route_unpack_inside_group(self, tuples):
        forwarding_table = self.generate_forwarding_table_unpack_inside_group()

        # 生成空路由表
        route_table = list()
        for i in range(self.N):
            route_table.append([0] * self.N)

        start_time = time.time()
        for dcu_out in tuples:
            dcu_out_idx = dcu_out['idx']
            row, col = dcu_out_idx % self.number_of_groups, dcu_out_idx // self.number_of_groups

            for dcu_in in tuples:  # 对于任意两个dcu生成路由表
                dcu_in_idx = dcu_in['idx']

                if dcu_out_idx % self.number_of_groups == dcu_in_idx % self.number_of_groups:
                    route_table[dcu_out_idx][dcu_in_idx] = dcu_out_idx
                elif dcu_in_idx in forwarding_table[row][col]:
                    route_table[dcu_out_idx][dcu_in_idx] = dcu_out_idx
                else:
                    dst = -1
                    for i in range(self.n_gpu_per_group):
                        if dcu_in_idx in forwarding_table[row][i]:
                            dst = self.number_of_groups * i + row
                    assert dst != -1
                    route_table[dcu_out_idx][dcu_in_idx] = dst

            # self.show_progress(dcu_out_idx, self.N, start_time)

        return route_table

    # group内拆包，流量通过采样的方式计算
    def generate_route_unpack_inside_group_sampling(self, tuples):
        forwarding_table = self.generate_forwarding_table_unpack_inside_group_with_sampling()

        # 生成空路由表
        route_table = list()
        for i in range(self.N):
            route_table.append([0] * self.N)

        # 根据转发表生成路由表
        start_time = time.time()
        for dcu_out in tuples:
            dcu_out_idx = dcu_out['idx']
            row, col = dcu_out_idx % self.number_of_groups, dcu_out_idx // self.number_of_groups

            for dcu_in in tuples:  # 对于任意两个dcu生成路由表
                dcu_in_idx = dcu_in['idx']

                if dcu_out_idx % self.number_of_groups == dcu_in_idx % self.number_of_groups:
                    route_table[dcu_out_idx][dcu_in_idx] = dcu_out_idx
                elif dcu_in_idx in forwarding_table[row][col]:
                    route_table[dcu_out_idx][dcu_in_idx] = dcu_out_idx
                else:
                    dst = -1
                    for i in range(self.n_gpu_per_group):
                        if dcu_in_idx in forwarding_table[row][i]:
                            dst = self.number_of_groups * i + row
                    assert dst != -1
                    route_table[dcu_out_idx][dcu_in_idx] = dst

            self.show_progress(dcu_out_idx, self.N, start_time)

        return route_table

    def generate_route_17280_v1(self, tuples):
        """
        group:
        [0, 128, 256, ..., 17152]
        [1, 129, 257, ..., 17153]
        ...
        [127, 255, 383, ..., 17279]
        """
        # import pickle
        # if not os.path.exists(self.route_path + 'forwarding_table.pickle'):
        #     self.generate_forwarding_table_17280(self.number_of_groups, self.n_gpu_per_group,
        #                                          max_link=int(1.2*np.sqrt(self.N)), max_rate=1.15)
        #
        # with open(self.route_path + 'forwarding_table.pickle', 'rb') as f:
        #     forwarding_table = pickle.load(f)

        forwarding_table = self.generate_forwarding_table_17280(self.number_of_groups, self.n_gpu_per_group,
                                                                max_link=int(1.2*np.sqrt(self.N)), max_rate=1.15)

        # 生成空路由表
        route_table = np.zeros((self.N, self.N), dtype=int)

        # 根据转发表生成路由表
        start_time = time.time()
        for dcu_out in tuples:
            dcu_out_idx = dcu_out['idx']
            row, col = dcu_out_idx % self.n_gpu_per_group, dcu_out_idx // self.n_gpu_per_group
            # time1 = time.time()
            for dcu_in in tuples:  # 对于任意两个dcu生成路由表
                dcu_in_idx = dcu_in['idx']

                if dcu_out_idx % self.n_gpu_per_group == dcu_in_idx % self.n_gpu_per_group:  # 在同一个group
                    route_table[dcu_out_idx][dcu_in_idx] = dcu_out_idx
                elif dcu_in_idx in forwarding_table[row][col]:  # 是本dcu负责转发的
                    route_table[dcu_out_idx][dcu_in_idx] = dcu_out_idx
                else:  # 找到负责转发的
                    dst = -1
                    for i in range(self.number_of_groups):
                        if dcu_in_idx in forwarding_table[row][i]:
                            dst = self.n_gpu_per_group * i + row
                    assert dst != -1
                    route_table[dcu_out_idx][dcu_in_idx] = dst
                    # print(dcu_in_idx)
            # time2 = time.time()
            # print('%.2f consumed.' % (time2 - time1))
                self.show_progress(dcu_out_idx*self.N + dcu_in_idx, self.N*self.N, start_time)

        return route_table

    def generate_route_default_1dim(self, N):
        route_name = 'route_default_1dim_' + str(N)
        npy_saving_path = self.root_path + 'tables/route_table/' + route_name + '.npy'
        json_saving_path = self.root_path + 'tables/route_table/' + route_name + '.json'

        route_table = list()
        for i in range(self.N):
            route_table.append([i] * self.N)
        route_table = np.array(route_table)

        np.save(npy_saving_path, route_table)
        print(route_name + '.npy saved')
        self.save_route_json(route_table, json_saving_path)

        return route_table

    def generate_route_default_2dim(self, N, number_of_group, dcu_per_group):
        route_name = 'route_default_2dim_%d_%d' % (number_of_group, dcu_per_group)
        npy_saving_path = self.root_path + 'tables/route_table/' + route_name + '.npy'
        json_saving_path = self.root_path + 'tables/route_table/' + route_name + '.json'

        # if os.path.exists(self.route_path + 'route_demo_%d_%d.npy' % (dcu_per_group, number_of_group)):
        #     return np.load(self.route_path + 'route_demo_%d_%d.npy' % (dcu_per_group, number_of_group))

        assert number_of_group * dcu_per_group == N

        route_table = list()
        for i in range(self.N):
            route_table.append([0] * self.N)
        route_table = np.array(route_table)

        # 如果self.N = 17280，运行时间约为2分钟
        for i in range(N):
            for j in range(N):
                # 如果是组内，或者编号相对应，直接发送
                if i // dcu_per_group == j // dcu_per_group or i % dcu_per_group == j % dcu_per_group:
                    route_table[i][j] = i
                else:
                    route_table[i][j] = j // dcu_per_group * dcu_per_group + i % dcu_per_group

        np.save(npy_saving_path, route_table)
        print(route_name + '.npy saved.')
        self.save_route_json(route_table, json_saving_path)
        return route_table


if __name__ == "__main__":
    g = GenerateRoute()
    g.generate_route_default_1dim(N=1200)
    g.generate_route_default_2dim(N=1200, number_of_group=30, dcu_per_group=40)
    # g.generate_route(method='17280_v1')
    pass
