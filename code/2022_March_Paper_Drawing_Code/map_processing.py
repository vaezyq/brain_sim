import numpy as np

map_table = np.load("map_table/map_1200_v1_cortical_v2_without_invalid_idx.pkl", allow_pickle=True)

# 读入.pkl格式的映射表，以list的形式返回
def read_map_pkl(path):
    N = 1200
    import pickle
    f = open(path, 'rb')
    map_table = pickle.load(f)

    map_list = list()
    for i in range(N):
        map_list.append(map_table[str(i)])
    return map_list

    # 将dcu-体素映射表保存为pkl文件


def save_map_pkl(map_table, path):
    # 将映射表转化为字典的格式
    N = 1200
    map_pkl = dict()
    for i in range(N):
        map_pkl[str(i)] = map_table[i]

    import pickle
    with open(path, 'wb') as f:
        pickle.dump(map_pkl, f)

    print('%s saved.' % path)


def map_table_split_to_normal(map_table):
    map_table_split = list()
    for i in range(1200):
        map_table_split.append(list())
    # 生成按标号顺序将标号顺序映射到GPU的映射表，此时每个体素的初始部分都为1
    for gpu_idx in range(1200):
        for population_idx in map_table[gpu_idx].keys():
            map_table_split[gpu_idx].append(population_idx)

    return map_table_split


def map_table_transfer(map_table):
    origin_size = np.load("map_table/size.npy")
    print(len(origin_size))
    mp_171452_to_226030 = list()
    for i in range(len(origin_size)):
        if origin_size[i] != 0:
            mp_171452_to_226030.append(i)

    for i in range(len(map_table)):
        for j in range(len(map_table[i])):
            map_table[i][j] = mp_171452_to_226030[map_table[i][j]]

    return map_table


def test():
    path = "map_table/map_1200_v2_cortical_v2_without_invalid_idx.pkl"
    new_path = "map_table/map_1200_v2_cortical_v2.pkl"
    map_table_split = read_map_pkl(path)
    new_map = map_table_split_to_normal(map_table_split)
    new_map = map_table_transfer(new_map)
    save_map_pkl(new_map, new_path)


test()