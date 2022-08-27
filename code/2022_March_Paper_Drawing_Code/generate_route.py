import numpy as np

import time


def transform_coordinate(coordinate_1d: int,
                         dimensions: tuple):
    coordinates = []
    for dimension in reversed(dimensions):
        coordinates.append(coordinate_1d % dimension)
        coordinate_1d = int(coordinate_1d / dimension)
    return coordinates[::-1]


# 计算原始进程编号
def get_rank_num(rankCor, dimCor: tuple):
    index = len(rankCor) - 1
    rankNum = 0
    count = 1
    while index >= 0:
        rankNum += rankCor[index] * count
        count *= dimCor[index]
        index -= 1
    return rankNum


def generate_specifc_dim_route_default(N, dimensionSize):
    """
    :param N: 进程个数
    :param dimensionSize: 生成的维度（每个维度包含的坐标个数）
    :return: 以file_name的形式保存生成的npy文件,并会返回生成的路由表
    """
    dimensions = len(dimensionSize)
    rankCoordinate = np.zeros((N, dimensions))
    for i in range(N):
        rankCoordinate[i, :] = transform_coordinate(i, dimensionSize)

    stage_route = np.zeros((N, N))
    for in_rank in range(N):
        print(in_rank)
        in_rank_cor = np.zeros((dimensions,))
        in_rank_cor[:] = rankCoordinate[in_rank]
        out_rank_cor = np.zeros((dimensions,))
        # in_rank_cor = rankCoordinate[in_rank]
        for out_rank in range(N):

            in_rank_cor_tmp = np.zeros((dimensions,))
            in_rank_cor_tmp[:] = in_rank_cor
            # print(out_rank)
            if out_rank == in_rank:
                stage_route[in_rank][out_rank] = in_rank
                continue
            out_rank_cor = rankCoordinate[out_rank]
            index = 0

            while index < len(out_rank_cor):

                if in_rank_cor_tmp[index] == out_rank_cor[index]:
                    index += 1
                    continue
                else:
                    in_rank_cor_tmp[index] = out_rank_cor[index]
                    tmp = get_rank_num(in_rank_cor_tmp, dimCor=dimensionSize)
                    if tmp == out_rank:
                        stage_route[in_rank][out_rank] = in_rank
                    else:
                        stage_route[in_rank][out_rank] = tmp
                    break
    stage_route = stage_route.astype(int)
    file_name = 'route_default_' + str(dimensions) + 'dim'
    for i in range(dimensions):
        file_name = file_name + '_' + str(dimensionSize[i])
    file_name = file_name + '.npy'
    np.save('route_table' + '/' + file_name, stage_route)
    print('route_table' + '/' + file_name + ' saved.')
    return stage_route


def get_route_dict_out(route_table, out_idx, in_idx_list):
    route_dict_out_idx = {}
    for idx in in_idx_list:
        if idx == out_idx:
            continue
        else:
            # 直接发送
            if route_table[out_idx][idx] == out_idx and not (idx in route_dict_out_idx):
                route_dict_out_idx[idx] = {idx}
            else:
                if not (route_table[out_idx][idx] in route_dict_out_idx):
                    route_dict_out_idx[route_table[out_idx][idx]] = {route_table[out_idx][idx]}
                route_dict_out_idx[route_table[out_idx][idx]].add(idx)

    return route_dict_out_idx


def save_route_json(route_table, route_saving_path):
    """
    N = 10000: 20 seconds + 3 minutes needed.
    """
    N = route_table.shape[0]
    route_table = route_table.tolist()

    # 把路由表中自己发给自己的部分去掉
    new_route = list()
    for i in range(N):
        del route_table[i][i]
        new_route.append(route_table[i])

    # 将路由表转化为字典的格式，以便保存为json文件
    start_time = time.time()
    route_dic = dict()
    for i in range(N):
        print(i)
        route_dic[str(i)] = dict()

        # 生成src
        route_dic[str(i)]['src'] = new_route[i]

        # 生成dst
        dst = list()
        for j in range(N):
            if j != i:
                dst.append(j)
        route_dic[str(i)]['dst'] = dst
        # print(i)
        # self.show_progress(i, self.N, start_time)

    # 将路由表保存为json文件
    import json
    route_json = json.dumps(route_dic, indent=2, sort_keys=False)
    with open(route_saving_path, 'w') as json_file:
        json_file.write(route_json)

    end_time = time.time()
    print('\n%s saved. %2.fs consumed.' % (route_saving_path, (end_time - start_time)))


def confirm_route_table(route_table):
    route_table = np.array(route_table).astype(int)
    print('Begin to confirm route table...')
    start_time = time.time()
    N = route_table.shape[0]
    # 计算src要经过几次转发可以到达dst，同时判断dcu之间是否全部连通
    step_length = np.zeros((N, N), dtype=np.int)
    for src in range(N):
        for dst in range(N):
            # 判断是否存在通路，如果不存在通路，会陷入死循环
            temp_src = src
            while route_table[temp_src][dst] != temp_src:
                temp_src = route_table[temp_src][dst]
                step_length[src][dst] += 1
                assert step_length[src][dst] < 10
        if src % 1000 == 0:
            print('%d / %d' % (src, N))
        # show_progress(src, N, start_time)

    print()
    for i in range(np.max(step_length) + 1):
        print('转发次数为%d的频数： %d' % (i, int(np.sum(step_length == i))))
    print('频数总和：%d' % (N ** 2))



