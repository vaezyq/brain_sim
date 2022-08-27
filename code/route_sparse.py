import numpy as np
import pickle
import time
import os.path
from route_analysis import RouteAnalysis


class RouteSparse(RouteAnalysis):
    def __init__(self):
        super().__init__()

        self.route_sparse_pkl = None
        if os.path.exists(self.route_path + 'route_sparse.pkl'):
            with open(self.route_path + 'route_sparse.pkl', 'rb') as f:
                self.route_sparse_pkl = pickle.load(f)

        self.new_connection_path = self.route_path + 'new_connection_2.pkl'
        self.route_table = self.read_route_npy(self.route_npy_save_path)

    # 计算pkl格式的稀疏路由表的连接数
    def cal_genuine_link_number_base_sparse_route_table(self):
        with open(self.route_path + 'route_sparse.pkl', 'rb') as f:
            sparse_route_table = pickle.load(f)

        between_group_link_num = np.zeros(self.N, dtype=int)
        link_num = np.zeros(self.N, dtype=int)

        for i in range(self.N):
            link_num[i] = len(sparse_route_table[i][i])
            for key in sparse_route_table[i]:
                if len(sparse_route_table[i][key]) != 0:
                    between_group_link_num[i] += 1
            if i % 1000 == 0:
                print(i, '/', self.N)

        inside_group_link_num = link_num - between_group_link_num

        print('inside group:', np.max(inside_group_link_num), np.min(inside_group_link_num),
              np.average(inside_group_link_num))
        print('between group:', np.max(between_group_link_num), np.min(between_group_link_num),
              np.average(between_group_link_num))
        print('sum:', np.max(link_num), np.min(link_num), np.average(link_num))

    # need about 20 minutes when N = 16000
    def route_npy_to_route_sparse_pkl(self):
        traffic_table_base_dcu = np.load(self.traffic_table_base_dcu_path)
        binary_connection_table_base_dcu = np.array(traffic_table_base_dcu, dtype=bool)
        route_table = np.load(self.route_path + 'route.npy')
        sparse_route_table = list()

        start_time = time.time()
        for i in range(self.N):
            single_sparse_route_table = dict()

            # 先遍历一遍，找出除了自己之外的转发dcu负责到哪些dcu的转发
            for j in range(self.N):
                key = route_table[i][j]
                if key not in single_sparse_route_table:
                    single_sparse_route_table[key] = list()

                if binary_connection_table_base_dcu[j][i] == 1 and key != i:
                    single_sparse_route_table[key].append(j)

            # 再遍历一遍
            for j in range(self.N):
                key = route_table[i][j]
                if key == i:
                    if i == j:
                        single_sparse_route_table[key].append(j)
                    elif j not in single_sparse_route_table and binary_connection_table_base_dcu[j][i]:  # 有连接的组内连接
                        single_sparse_route_table[key].append(j)
                    elif j in single_sparse_route_table and len(single_sparse_route_table[j]) != 0:  # 组间转发不为空的组间连接
                        single_sparse_route_table[key].append(j)

            sparse_route_table.append(single_sparse_route_table)

            if i % 1000 == 0:
                print('%d / %d' % (i, self.N))
        end_time = time.time()
        print('%.2f seconds consumed.' % (end_time - start_time))

        with open(self.route_path + 'route_sparse.pkl', 'wb') as f:
            pickle.dump(sparse_route_table, f)
            print(self.route_path + 'route_sparse.pkl saved.')

    def update_route_sparse_pkl_with_dti(self):
        with open(self.new_connection_path, 'rb') as f:
            updated_route_in = pickle.load(f)

        updated_route_out = [[] for _ in range(self.N)]
        for dst in range(self.N):
            for src in updated_route_in[dst]:
                updated_route_out[src].append(dst)

        # 生成补全后的稀疏路由表 N=8000 需要约8分钟
        time1 = time.time()
        new_route_sparse_pkl = self.route_sparse_pkl.copy()
        for src in range(self.N):
            old_dsts = list()
            for key in self.route_sparse_pkl[src]:
                for dst in self.route_sparse_pkl[src][key]:
                    old_dsts.append(dst)
            old_dsts.sort()

            for dst in updated_route_out[src]:
                if dst not in old_dsts:
                    new_route_sparse_pkl[src][src].append(dst)

            if src % 1000 == 0:
                print('%d / %d' % (src, self.N))
        time2 = time.time()
        print('%.4f seconds consumed.' % (time2 - time1))

        with open(self.route_path + 'updated_route_sparse.pkl', 'wb') as f:
            pickle.dump(new_route_sparse_pkl, f)

        print(self.route_path + 'updated_route_sparse.pkl saved.')

    # 把pkl格式的路由表转化为json格式并存储，N=8000需要约1分钟
    def route_sparse_pkl_to_route_sparse_json(self, path):
        with open(path, 'rb') as f:
            route_sparse_pkl = pickle.load(f)

        start_time = time.time()
        route_dic = dict()
        for src in range(self.N):
            route_dic[str(src)] = dict()
            src_list = list()
            dst_list = list()

            for bridge in route_sparse_pkl[src]:
                for dst in route_sparse_pkl[src][bridge]:
                    if bridge != dst:
                        src_list.append(int(bridge))
                        dst_list.append(int(dst))

            route_dic[str(src)]['src'] = src_list
            route_dic[str(src)]['dst'] = dst_list

            if (src + 1) % 400 == 0:
                print('%d / %d' % (src + 1, self.N))

        import json
        route_json = json.dumps(route_dic, indent=2, sort_keys=False)
        with open(self.route_path + 'route_sparse.json', 'w') as json_file:
            json_file.write(route_json)

        end_time = time.time()
        print('%s saved. %2.fs consumed.' % (self.route_path + 'route_sparse.json', (end_time - start_time)))

    # 计算未更新debug神经元的稀疏路由表的连接数
    def cal_connection_number_per_dcu_base_route_sparse_pkl(self):
        connection_num_per_dcu_old_out = np.zeros(self.N)
        connection_num_per_dcu_old_in = np.zeros(self.N)
        for i in range(self.N):
            for key in self.route_sparse_pkl[i]:
                connection_num_per_dcu_old_out[i] += len(self.route_sparse_pkl[i][key])
                for dst in self.route_sparse_pkl[i][key]:
                    connection_num_per_dcu_old_in[dst] += 1

        print('Old route_sparse.pkl connection_out:', np.min(connection_num_per_dcu_old_out),
              np.max(connection_num_per_dcu_old_out),
              np.average(connection_num_per_dcu_old_out))

        print('Old route_sparse.pkl connection_in:', np.min(connection_num_per_dcu_old_in),
              np.max(connection_num_per_dcu_old_in),
              np.average(connection_num_per_dcu_old_in))

        return connection_num_per_dcu_old_in, connection_num_per_dcu_old_out

    # 计算更新debug神经元后稀疏路由表的连接数
    def cal_connection_number_per_dcu_base_updated_route_sparse_pkl(self):
        with open(self.route_path + 'updated_route_sparse.pkl', 'rb') as f:
            updated_route_sparse_pkl = pickle.load(f)
        connection_num_per_dcu_old_out = np.zeros(self.N)
        connection_num_per_dcu_old_in = np.zeros(self.N)
        for i in range(self.N):
            for key in updated_route_sparse_pkl[i]:
                connection_num_per_dcu_old_out[i] += len(updated_route_sparse_pkl[i][key])
                for dst in updated_route_sparse_pkl[i][key]:
                    connection_num_per_dcu_old_in[dst] += 1

        print('Updated route_sparse.pkl connection_out:', np.min(connection_num_per_dcu_old_out),
              np.max(connection_num_per_dcu_old_out),
              np.average(connection_num_per_dcu_old_out))

        print('Updated route_sparse.pkl connection_in:', np.min(connection_num_per_dcu_old_in),
              np.max(connection_num_per_dcu_old_in),
              np.average(connection_num_per_dcu_old_in))

        return connection_num_per_dcu_old_in, connection_num_per_dcu_old_out

    # 计算流量表的连接数
    def cal_connection_number_per_dcu_base_traffic_table(self):
        # 用dcu之间流量表算出来的连接数
        binary_traffic_table = np.array(self.traffic_base_dcu, dtype=bool)
        connection_check_out = np.zeros(self.N)
        connection_check_in = np.zeros(self.N)

        for i in range(self.N):
            connection_check_out[i] = np.sum(binary_traffic_table[:, i])
            connection_check_in[i] = np.sum(binary_traffic_table[i, :])

        print('Check_out:', np.min(connection_check_out),
              np.max(connection_check_out),
              np.average(connection_check_out))
        print('Check_in:', np.min(connection_check_in),
              np.max(connection_check_in),
              np.average(connection_check_in))

        return connection_check_in, connection_check_out

    # 检查new_connection中的连接数与更新后的稀疏路由表是否一致
    def check_update_route_sparse_pkl_and_new_connection(self):
        with open(self.new_connection_path, 'rb') as f:
            updated_route_in = pickle.load(f)

        with open(self.route_path + 'updated_route_sparse.pkl', 'rb') as f:
            updated_route_sparse_pkl = pickle.load(f)

        connection_list = [[] for _ in range(self.N)]
        for i in range(self.N):
            for bridge in updated_route_sparse_pkl[i]:
                for dst in updated_route_sparse_pkl[i][bridge]:
                    connection_list[i].append(dst)
            connection_list[i].sort()

        for dst in range(self.N):
            for src in updated_route_in[dst]:
                assert dst in connection_list[src]

            if dst % 1000 == 0:
                print('%d / %d' % (dst, self.N))

        print('Nice')

    # 将npy格式的路由表转化为pkl格式的稀疏路由表，根据dti数据更新pkl路由表，并保存为json格式
    def update_check_save_to_json(self):
        print('begin save json...')
        self.save_route_json(self.route_table, self.route_dense_json_save_path)
        print('begin save sparse...')
        self.route_npy_to_route_sparse_pkl()
        # self.update_route_sparse_pkl_with_dti()
        # self.route_sparse_pkl_to_route_sparse_json(self.route_path + 'updated_route_sparse.pkl')

    def check(self):
        a, updated_out = self.cal_connection_number_per_dcu_base_updated_route_sparse_pkl()
        b, original_out = self.cal_connection_number_per_dcu_base_traffic_table()
        c, old_out = self.cal_connection_number_per_dcu_base_route_sparse_pkl()

        for i in range(self.N):
            if old_out[i] > updated_out[i]:
                print('error!')
                break


if __name__ == '__main__':
    job = RouteSparse()
    job.update_check_save_to_json()
