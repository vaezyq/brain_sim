# 类脑提供的代码，用于统计模拟得到的slurm文件中，打印每个阶段执行的时间

import numpy as np
import re
import os

iter = 1000
n = 2000
# n = 10000
# n = 1200
# n = 2000
data1 = []
data2 = []
data3 = []
data4 = []
data5 = []
data6 = []
data7 = []
data8 = []
data9 = []
data10 = []
data11 = []
# data11 = np.zeros((m, n))
# print(data1.dtype())
# path = r'output_1001.log'
# path = 'tools_cpytest_80m'
path = 'route_table/'
files = os.listdir(path)
# print(files)

pattern1 = re.compile(r'send duration:')
pattern2 = re.compile(r'recv duration:')
pattern3 = re.compile(r'route duration:')
pattern4 = re.compile(r'compute duration:')
pattern5 = re.compile(r'report duration:')
pattern6 = re.compile(r'copy duration before send:')
pattern7 = re.compile(r'copy duration after recv:')
pattern8 = re.compile(r'parse merge duration:')
pattern9 = re.compile(r'compute duration before route:')
pattern10 = re.compile(r'copy duration before report:')
# pattern11 = re.compile(r'iter duration:')
pattern11 = re.compile(r'copy count after recv:')

for file in files:
    # if (re.match('slurm-12074362', file) == None): #16route
    # if (re.match('slurm-12076338', file) == None): #16p2p
    # if (re.match('slurm-12084595', file) == None): #400p2p
    # if (re.match('slurm-12094766', file) == None): #16p2p  12095171
    # if (re.match('slurm-12095174', file) == None): #16p2pnofreq
    # if (re.match('slurm-12202511', file) == None): #16p2p
    # if (re.match('slurm-12207139', file) == None): #16route
    # if (re.match('slurm-12212736', file) == None): #400p2p
    # if (re.match('slurm-12322784', file) == None): #400route
    # if (re.match('slurm-12363742', file) == None): #400route
    # if (re.match('slurm-12383499', file) == None): #400p2ppairwise
    # if (re.match('slurm-12384809', file) == None): #400p2prandom
    # if (re.match('slurm-12384906', file) == None): #400routepairwise
    # if (re.match('slurm-12384982', file) == None): #400routerandom
    # if (re.match('slurm-12418233', file) == None): #1200p2p
    # if (re.match('slurm-12419305', file) == None): #1200route
    # if (re.match('slurm-12420306', file) == None): #1200route2
    # if (re.match('slurm-12447296', file) == None): #2000p2p
    # if (re.match('slurm-12447431', file) == None): #2000route
    # if (re.match('slurm-12453981', file) == None): #2000routepair.log
    # if (re.match('slurm-12447583', file) == None): #2000routerandom.log
    # if (re.match('slurm-14337782', file) == None): #10000route.log
    if (re.match('slurm-16976483.out', file) == None):  # 16route.log
        continue
    position = path + '/' + file
    print(position)
    with open(position, 'r', encoding='utf-8') as f:
        for line in f:
            value = line[:-1]  # 去掉换行符
            if pattern1.match(value):
                # data.append(value.split(':')[-1])
                tmp = value.split(':')[-1]
                tmp = float(tmp)
                # tmp = tmp[0:-9]
                data1.append(float(tmp))
            if pattern2.match(value):
                # data.append(value.split(':')[-1])
                tmp = value.split(':')[-1]
                tmp = float(tmp)
                # print('revc duration', tmp)
                data2.append(tmp)
                # np.save('cpy', data2)
            if pattern3.match(value):
                tmp = value.split(':')[-1]
                tmp = float(tmp)
                data3.append(float(tmp))
            if pattern4.match(value):
                tmp = value.split(':')[-1]
                tmp = float(tmp)
                data4.append(float(tmp))
            if pattern5.match(value):
                tmp = value.split(':')[-1]
                tmp = float(tmp)
                data5.append(float(tmp))
            if pattern6.match(value):
                tmp = value.split(':')[-1]
                tmp = float(tmp)
                data6.append(float(tmp))
            if pattern7.match(value):
                tmp = value.split(':')[-1]
                tmp = float(tmp)
                data7.append(float(tmp))
            if pattern8.match(value):
                tmp = value.split(':')[-1]
                tmp = float(tmp)
                data8.append(float(tmp))
            if pattern9.match(value):
                tmp = value.split(':')[-1]
                tmp = float(tmp)
                data9.append(float(tmp))
            if pattern10.match(value):
                tmp = value.split(':')[-1]
                tmp = float(tmp)
                data10.append(float(tmp))
            if pattern11.match(value):
                tmp = value.split(':')[-1]
                tmp = float(tmp)
                data11.append(float(tmp))
'''
pattern1 = re.compile(r'send duration:')
pattern2 = re.compile(r'recv duration:')
pattern3 = re.compile(r'route duration:')
pattern4 = re.compile(r'compute duration:')
pattern5 = re.compile(r'report duration:')
pattern6 = re.compile(r'copy duration before send:')
pattern7 = re.compile(r'copy duration after recv:')
pattern8 = re.compile(r'parse merge duration:')
pattern9 = re.compile(r'compute duration before route:')
pattern10 = re.compile(r'copy duration before report:')
pattern11 = re.compile(r'iter duration:')				
'''
# copy_before_sending_duration  pattern6
print('copy duration before send mean')
print('std')
print('max mean')
print('argmax.mode')
print('cov')
print('std.argmax')

# sending_duration  pattern1
print('send duration mean')
print('std')
print('max mean')
print('argmax.mode')
print('cov')
print('std.argmax')

# parse_merge_duration   pattern8
print('parse merge duration mean')
print('std')
print('max mean')
print('argmax.mode')
print('cov')
print('std.argmax')

# pattern3
print('route duration mean')
print('std')
print('max mean')
print('argmax.mode')
print('cov')
print('std.argmax')

# route_computing_duration  pattern9
print('compute duration before route mean')
print('std')
print('max mean')
print('argmax.mode')
print('cov')
print('std.argmax')

# pattern2
print('recv duration mean')
print('std')
print('max mean')
print('argmax.mode')
print('cov')
print('std.argmax')

# copy_after_recving_duration  pattern7
print('copy_after_recving_duration_ mean')
print('std')
print('max mean')
print('argmax.mode')
print('cov')
print('std.argmax')

# computing_duration  pattern4
print('compute duration mean')
print('std')
print('max mean')
print('argmax.mode')
print('cov')
print('std.argmax')

# copy_before_reporting_duration  pattern10
print('copy duration before report mean')
print('std')
print('max mean')
print('argmax.mode')
print('cov')
print('std.argmax')

# pattern5
print('report duration mean')
print('std')
print('max mean')
print('argmax.mode')
print('cov')
print('std.argmax')

print('iter duration mean')
print('std')
print('max mean')
print('argmax.mode')
print('cov')
print('std.argmax')

# ata1 = data1.astype(np.float)
# print('size', data1.shape)
# print('send duration mean :', data1.mean()*1000, ' ms')
# print('send duration mean :\n', data1.mean())
# print('send duration std :\n', data1.std())
# np.save('mpi', data1)

# print('size :', data2.shape)
# print('recv duration mean :\n', data2.mean())
# print('revc duration std :\n', data2.std())


# print('size :', data3.shape)
# print('route duration mean :\n', data3.mean())
# print('route duration std :\n', data3.std())


# print('size :', data4.shape)
# print('compute duration mean :\n', data4.mean())
# print('compute duration std :\n', data4.std())


# print('size :', data5.shape)
# print('report duration mean :\n', data5.mean())
# print('report duration std :\n', data5.std())


# print('size :', data6.shape)
# print('copy duration before send mean :\n', data6.mean())
# print('copy duration before send std :\n', data6.std())


# print('size :', data7.shape)
# print('copy duration after recv mean :\n', data7.mean())
# print('copy duration after recv std :\n', data7.std())


# print('size :', data8.shape)
# print('parse merge duration mean :\n', data8.mean())
# print('parse merge duration std :\n', data8.std())


# print('size :', data9.shape)
# print('compute duration before route mean :\n', data9.mean())
# print('compute duration before route std :\n', data9.std())


# print('size :', data10.shape)
# print('copy duration before report mean :\n', data10.mean())
# print('copy duration before report std :\n', data10.std())

print('Copy duration before send:')
np.save('copy duration before send', data6)
data6 = np.load('copy duration before send.npy')
print(data6.mean() * 1000)
print(data6.std() * 1000)
print(data6.reshape((iter, n)).max(axis=1).mean() * 1000)
print(np.bincount(data6.reshape((iter, n)).argmax(axis=1)).argmax())
print(data6.reshape((iter, n)).std(axis=0).max() * 1000)
print(data6.reshape((iter, n)).std(axis=0).argmax())

print('Send duration:')
np.save('send duration', data1)
# np.save('recv duration', data2)
data1 = np.load('send duration.npy')
print(data1.mean() * 1000)
print(data1.std() * 1000)
print(data1.reshape((iter, n)).max(axis=1).mean() * 1000)
print(np.bincount(data1.reshape((iter, n)).argmax(axis=1)).argmax())
print(data1.reshape((iter, n)).std(axis=0).max() * 1000)
print(data1.reshape((iter, n)).std(axis=0).argmax())

print('Parse merge duration:')
np.save('parse merge duration', data8)
data8 = np.load('parse merge duration.npy')
print(data8.mean() * 1000)
print(data8.std() * 1000)
print(data8.reshape((iter, n)).max(axis=1).mean() * 1000)
print(np.bincount(data8.reshape((iter, n)).argmax(axis=1)).argmax())
print(data8.reshape((iter, n)).std(axis=0).max() * 1000)
print(data8.reshape((iter, n)).std(axis=0).argmax())

print('Route duration:')
np.save('route duration', data3)
data3 = np.load('route duration.npy')
print(data3.mean() * 1000)
print(data3.std() * 1000)
print(data3.reshape((iter, n)).max(axis=1).mean() * 1000)
print(np.bincount(data3.reshape((iter, n)).argmax(axis=1)).argmax())
print(data3.reshape((iter, n)).std(axis=0).max() * 1000)
print(data3.reshape((iter, n)).std(axis=0).argmax())

print('Compute duration before route:')
np.save('compute duration before route', data9)
data9 = np.load('compute duration before route.npy')
print(data9.mean() * 1000)
print(data9.std() * 1000)
print(data9.reshape((iter, n)).max(axis=1).mean() * 1000)
print(np.bincount(data9.reshape((iter, n)).argmax(axis=1)).argmax())
print(data9.reshape((iter, n)).std(axis=0).max() * 1000)
print(data9.reshape((iter, n)).std(axis=0).argmax())

print('Recv duration:')
np.save('recv duration', data2)
data2 = np.load('recv duration.npy')
print(data2.mean() * 1000)
print(data2.std() * 1000)
print(data2.reshape((iter, n)).max(axis=1).mean() * 1000)
print(np.bincount(data2.reshape((iter, n)).argmax(axis=1)).argmax())
print(data2.reshape((iter, n)).std(axis=0).max() * 1000)
print(data2.reshape((iter, n)).std(axis=0).argmax())

print('Copy duration after recv:')
np.save('copy duration after recv', data7)
data7 = np.load('copy duration after recv.npy')
print(data7.mean() * 1000)
print(data7.std() * 1000)
print(data7.reshape((iter, n)).max(axis=1).mean() * 1000)
print(np.bincount(data7.reshape((iter, n)).argmax(axis=1)).argmax())
print(data7.reshape((iter, n)).std(axis=0).max() * 1000)
print(data7.reshape((iter, n)).std(axis=0).argmax())

print('Compute duration:')
np.save('compute duration', data4)
data4 = np.load('compute duration.npy')
print(data4.mean() * 1000)
print(data4.std() * 1000)
print(data4.reshape((iter, n)).max(axis=1).mean() * 1000)
print(np.bincount(data4.reshape((iter, n)).argmax(axis=1)).argmax())
print(data4.reshape((iter, n)).std(axis=0).max() * 1000)
print(data4.reshape((iter, n)).std(axis=0).argmax())

print("Copy duration before report:")
np.save('copy duration before report', data10)
data10 = np.load('copy duration before report.npy')
print(data10.mean() * 1000)
print(data10.std() * 1000)
print(data10.reshape((iter, n)).max(axis=1).mean() * 1000)
print(np.bincount(data10.reshape((iter, n)).argmax(axis=1)).argmax())
print(data10.reshape((iter, n)).std(axis=0).max() * 1000)
print(data10.reshape((iter, n)).std(axis=0).argmax())

print("Report duration:")
np.save('report duration', data5)
data5 = np.load('report duration.npy')
print(data5.mean() * 1000)
print(data5.std() * 1000)
print(data5.reshape((iter, n)).max(axis=1).mean() * 1000)
print(np.bincount(data5.reshape((iter, n)).argmax(axis=1)).argmax())
print(data5.reshape((iter, n)).std(axis=0).max() * 1000)
print(data5.reshape((iter, n)).std(axis=0).argmax())

print("Iter duration:")
np.save('iter duration', data11)
data11 = np.load('iter duration.npy')
print(data11.mean() * 1000)
print(data11.std() * 1000)
print(data11.reshape((iter, n)).max(axis=1).mean() * 1000)
print(np.bincount(data11.reshape((iter, n)).argmax(axis=1)).argmax())
print(data11.reshape((iter, n)).std(axis=0).max() * 1000)
print(data11.reshape((iter, n)).std(axis=0).argmax())
