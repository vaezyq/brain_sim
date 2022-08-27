# 对类脑提供的代码timetest.py的改进，统计模拟得到的slurm文件中，每个阶段执行的时间
# 用于将slurm文件里每个阶段执行的时间记录到excel中

import numpy as np
import re
import pandas as pd
import os


class TimeTest:
    def __init__(self):
        self.pattern = None
        self.iter = 1000
        self.n = 2000
        self.data_num = 11

        self.slurm_name = 'slurm-16976483.out'
        self.file_path = 'time_calculate' + '/' + self.slurm_name[:-3]
        if not os.path.exists(self.file_path):
            os.makedirs(self.file_path)
        self.data = np.zeros((self.data_num, self.iter * self.n))
        self.pattern_dict = self.get_pattern_dict()

    def get_pattern_dict(self):
        pattern_dict = {}
        self.pattern = ['copy duration before send:', 'send duration:', 'parse merge duration:', 'route duration:',
                        'compute duration before route:', 'recv duration:', 'copy duration after recv:'
            , 'compute duration:', 'copy duration before report:', 'report duration:', 'copy count after recv:']

        for i in range(self.data_num):
            pattern_dict.setdefault(self.pattern[i], i)
        return pattern_dict

    def get_dim(self, st):
        dim = -1
        for key, value in self.pattern_dict.items():
            if key in st:
                dim = value
        return dim

    def get_data(self):
        with open(self.slurm_name, 'r', encoding='utf-8') as f:
            index = [0] * 11
            count = 0
            for line in f:
                value = line[:-1]
                dim = self.get_dim(value)
                # print(dim)
                if dim != -1:
                    self.data[dim][index[dim]] = float(value.split(':')[-1])
                    index[dim] += 1
            np.save(self.file_path + '/' + "data.npy", self.data)
            print(self.file_path + '/' + "data.npy " + "saved")

    def calculate_data(self, index):
        data_tmp = np.zeros((6,))
        data_tmp[0] = self.data[index].mean() * self.iter
        data_tmp[1] = self.data[index].std() * self.iter
        data_tmp[2] = self.data[index].reshape((self.iter, self.n)).max(axis=1).mean() * self.iter
        data_tmp[3] = np.bincount(self.data[index].reshape((self.iter, self.n)).argmax(axis=1)).argmax()
        data_tmp[4] = self.data[index].reshape((self.iter, self.n)).std(axis=0).max() * self.iter
        data_tmp[5] = self.data[index].reshape((self.iter, self.n)).std(axis=0).argmax()
        return data_tmp

    def write_to_csv(self):
        # meaning_col = np.zeros((self.data_num * 6,), dtype=str)
        meaning_col = [''] * self.data_num * 6
        data_col = np.zeros((self.data_num * 6,))
        col_dict = {0: 0, 1: 'std', 2: 'max mean', 3: 'argmax.mode', 4: 'cov', 5: 'std.argmax'}
        for i in range(self.data_num * 6):
            num = i % 6
            tep = col_dict.get(num)
            if num == 0:
                if i != 60:
                    meaning_col[i] = self.pattern[int(i / 6)][:-1] + " mean"
                else:
                    meaning_col[i] = 'iter duration:'[:-1] + " mean"
            else:
                meaning_col[i] = tep
        data_col = np.zeros((self.data_num * 6,))
        for i in range(self.data_num):
            data_col[i * 6: (i + 1) * 6] = self.calculate_data(i)

        dataframe = pd.DataFrame({'meaning': meaning_col, 'data': data_col})

        dataframe.to_csv(self.file_path + '/' + "test.csv", index=False, sep=',')
        print(self.file_path + '/' + "test.csv" + "saved")


if __name__ == "__main__":
    time_test = TimeTest()

    time_test.get_data()
    time_test.write_to_csv()
