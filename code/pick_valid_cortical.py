"""
·本文件用于从226030*226030的功能柱级别的连接概率表中，筛选出size为0的功能柱，
 并将其从连接概率表中剔除，最终生成171452*171452的连接概率矩阵
"""

import os.path
import sparse
from sparse import COO
import pickle
import numpy as np
import time


class PreProcess:
    def __init__(self):
        self.conn_version_ = 'cortical_v2/'
        self.conn_root_ = '../tables/conn_table/' + self.conn_version_

        self.conn_table_path_ = self.conn_root_ + 'cortical_and_subcortical_conn_prob_22703.pickle'
        self.size_degree_path_ = self.conn_root_ + 'cortical_and_subcortical_size_info_22703.npz'

        self.old_size_path_ = self.conn_root_ + 'origin_size.npy'
        self.old_degree_path_ = self.conn_root_ + 'origin_degree.npy'

        self.new_conn_path_ = self.conn_root_ + 'conn.pickle'
        self.new_size_path_ = self.conn_root_ + 'size.npy'
        self.new_degree_path_ = self.conn_root_ + 'degree.npy'

        self.valid_idx_path_ = self.conn_root_ + 'valid_idx.npy'

        self.valid_idx_ = None
        self.n_ = None
        self.initialize_valid_idx()

    def initialize_valid_idx(self):
        if os.path.exists(self.valid_idx_path_):
            self.valid_idx_ = np.load(self.valid_idx_path_)
        else:
            data = np.load(self.size_degree_path_)
            size = data['size']

            self.valid_idx_ = np.array([], dtype=int)
            for i in range(size.shape[0]):
                if size[i] != 0:
                    self.valid_idx_ = np.append(self.valid_idx_, i)
            self.save_npy(self.valid_idx_path_, self.valid_idx_)

        self.n_ = self.valid_idx_.shape[0]
        print("Number of valid cortical: ", self.n_)

    @staticmethod
    def save_npy(file_path, file):
        if os.path.exists(file_path):
            np.save(file_path, file)
            print(file_path + ' saved. (covered)')
        else:
            np.save(file_path, file)
            print(file_path + ' saved.')

    def pick_valid_size_degree(self):
        data = np.load(self.size_degree_path_)
        size = data['size']
        degree = data['degree']

        self.save_npy(self.old_size_path_, size)
        self.save_npy(self.old_degree_path_, degree)

        print(size.shape)
        print(degree.shape)

        new_size = np.empty(self.n_)
        new_degree = np.empty(self.n_)
        for i in range(self.n_):
            old_idx = self.valid_idx_[i]
            new_size[i] = size[old_idx]
            new_degree[i] = degree[old_idx]

        self.save_npy(self.new_size_path_, new_size)
        self.save_npy(self.new_degree_path_, new_degree)

    def pick_valid_conn(self):
        print("\n##### Begin processing conn...")
        with open(self.conn_table_path_, 'rb') as f:
            conn = pickle.load(f)
        print(self.conn_table_path_ + " loaded.")

        new_conn = conn.tocsr()
        new_conn = new_conn.tolil()

        start_time = time.time()
        print("Begin indexing...")
        new_conn = new_conn[np.ix_(self.valid_idx_, self.valid_idx_)]
        end_time = time.time()
        print('%.2fs consumed.' % (end_time - start_time))

        print("lil shape:", new_conn.shape)

        new_conn = new_conn.tocoo()
        print("coo shape:", new_conn.shape)

        new_conn = COO.from_scipy_sparse(new_conn)
        print("sparse._coo shape:", new_conn.shape)

        with open(self.new_conn_path_, 'wb') as f:
            pickle.dump(new_conn, f)
        print(self.new_conn_path_ + " saved")

        print("old conn:", conn.nnz)
        print("new conn:", new_conn.nnz)
        print("non zero rate: %.6f" % (new_conn.nnz / self.n_ / self.n_))

    def confirm_conn(self):
        new_path = '../tables/conn_table/valid_cortical_and_subcortical_after_indexing.pickle'

        f = open(self.conn_table_path_, 'rb')
        conn = pickle.load(f)

        data = np.load(self.size_degree_path_)
        size = data['size']
        degree = data['degree']

        valid_idx = np.array([], dtype=int)
        for i in range(size.shape[0]):
            if size[i] != 0:
                valid_idx = np.append(valid_idx, i)

        f = open(new_path, 'rb')
        new_conn = pickle.load(f)

        row, col = 0, 0
        for row_idx in valid_idx:
            col = 0
            for col_idx in valid_idx:
                # print(row_idx, col_idx, conn[row_idx][col_idx], new_conn[row][col])
                if conn[row_idx][col_idx] != new_conn[row][col]:
                    return
                col += 1
            row += 1
            if row_idx % 1 == 0:
                print("=====#######  %d  #######=====" % row_idx)

    # when N = 10000, 30 minutes needed
    def preprocess(self):
        self.pick_valid_size_degree()
        self.pick_valid_conn()
        self.confirm_conn()


if __name__ == '__main__':
    Job = PreProcess()
    Job.preprocess()
