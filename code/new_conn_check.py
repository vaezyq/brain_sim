import numpy as np
import matplotlib.pyplot as plt
import pickle


path = "../tables/conn_table/cortical_v2/"

size_path = path + "size.npy"
degree_path = path + "degree.npy"
conn_path = path + "conn.pickle"

size = np.load(size_path)
degree = np.load(degree_path)
with open(conn_path, 'rb') as f:
    conn = pickle.load(f)

traffic_table_base_dcu_path = "../tables/traffic_table/traffic_table_base_dcu_map_10000_sequential_cortical_v2.npy"

data = np.load(traffic_table_base_dcu_path)
