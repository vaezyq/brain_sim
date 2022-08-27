import sparse
from sparse import COO
import pickle
import numpy as np
import time

conn = sparse.random((4, 4), density=1)
for i in range(4):
    for j in range(4):
        print(conn[i][j], end=' ')
    print()


valid_idx = np.array([1, 3])
new_conn = conn.tocsr()
new_conn = new_conn.tolil()

start_time = time.time()
print("Begin indexing...")
new_conn = new_conn[np.ix_(valid_idx, valid_idx)]
end_time = time.time()
print('%.2fs consumed.' % (end_time - start_time))

print("lil shape:", new_conn.shape)

new_conn = new_conn.tocoo()
print("coo shape:", new_conn.shape)

new_conn = COO.from_scipy_sparse(new_conn)
print("sparse._coo shape:", new_conn.shape)

file = open('../tables/conn_table/valid_cortical_and_subcortical_after_indexing.pickle', 'wb')
pickle.dump(new_conn, file)
file.close()

f = open('../tables/conn_table/valid_cortical_and_subcortical_after_indexing.pickle', 'rb')
data = pickle.load(f)
print()
for i in range(2):
    for j in range(2):
        print(data[i][j], end=' ')
    print()
