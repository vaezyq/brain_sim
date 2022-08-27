import numpy as np
from base import Base


class Data(Base):
    def __init__(self):
        super().__init__()

        # Biological data
        self.conn = None
        self.conn_dict = None
        self.size, self.origin_size = None, None
        self.degree, self.origin_degree = None, None
        self.n, self.n_origin = None, None
        self.load_data()

        self.size_multi_degree = np.multiply(self.size, self.degree)

    def load_conn(self):
        if self.conn_version[0:5] == 'voxel':
            self.conn = np.load(self.conn_table_path)
        else:
            # import pickle
            # import sparse
            # f = open(self.conn_root + 'conn.pickle', 'rb')
            # self.conn = pickle.load(f)
            self.conn_root='/public/home/ssct005t/project/wml_istbi/tables/conn_table/cortical_v2/'
            self.conn_dict = np.load(self.conn_root + 'conn_dict_int.npy', allow_pickle=True).item()

    def load_size(self):
        self.size = np.load(self.size_path)
        if self.conn_version[0:5] == 'voxel':
            self.origin_size = self.size
        else:
            self.origin_size = np.load(self.origin_size_path)

    def load_degree(self):
        if self.conn_version[0:5] == 'voxel':
            self.degree = np.array([100] * self.n)
            # self.origin_degree = self.degree
        else:
            self.degree = np.load(self.degree_path)
            self.origin_degree = np.load(self.origin_degree_path)

    def load_data(self):
        self.load_conn()
        self.load_size()
        self.n = self.size.shape[0]
        self.n_origin = self.origin_size.shape[0]
        self.load_degree()

    def show_basic_information(self):
        msg = ' Base Info '
        k_pounds = 19
        print('#' * k_pounds + msg + '#' * k_pounds)
        print("Number of GPUs used:", self.N)
        print("Number of voxels:", self.n)
        print("Number of neuron: %.2e" % self.neuron_number)
        print("Number of groups:", self.number_of_groups)
        print("Number of GPUs per group:", self.n_gpu_per_group)
        print("Connection table version:", self.conn_version)
        print("Map version:", self.map_version)
        print("Route version:", self.route_version)
        print("Sum size:", np.sum(self.size))
        print("Sum degree:", np.sum(self.degree))
        print('#' * (k_pounds * 2 + len(msg)))
        print()


if __name__ == '__main__':
    D = Data()
    pass
