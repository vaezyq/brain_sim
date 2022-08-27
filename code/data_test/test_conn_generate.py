import numpy as np

conn_dict_int = np.load("../conn_dict_int.npy", allow_pickle=True).item()

print("总字典长度")
print(len(conn_dict_int))

conn_dict_test = {}

conn_no_zero_index = np.load("conn_no_zero_index.npy", allow_pickle=True)
len_dict = len(conn_no_zero_index[0])
N = 171508
len_new = 2 * N
for i in range(len_dict):
    x = int(conn_no_zero_index[0][i])
    y = int(conn_no_zero_index[1][i])
    key = x * N + y

    key_1 = int(x * len_new + y)
    key_2 = int((x + N) * len_new + y)
    key_3 = int(x * len_new + (y + N))
    key_4 = int((x + N) * len_new + (y + N))

    conn_dict_test[key_1] = conn_dict_test[key_2] = conn_dict_test[key_3] = conn_dict_test[key_4] = conn_dict_int[
        key]
    # print(conn_dict_int[key])
    if i % 10000 == 1:
        print(i)

np.save("conn_dict_test.npy", conn_dict_test)
