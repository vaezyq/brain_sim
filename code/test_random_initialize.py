import unittest

import numpy as np
from mpi4py import MPI
from brain_block.random_initialize import *
from scipy.io import loadmat
from brain_block.block import block
import sparse
import pickle
import os


class TestBlock(unittest.TestCase):
    '''
    20Hz, max=100Hz, degree=1000
    (0.001617693924345076, 
    0.0001014432345982641, 
    0.003837576135993004, 
    0.000286647496977821)
    
    20Hz, max=100Hz, degree=100
    (0.010177472606301308, 
    0.0006282327813096344, 
    0.03813745081424713, 
    0.0021428221371024847)
    
    20Hz, max=100Hz, degree=137
    (0.011156645603477955, 
    0.0006994128925725818, 
    0.03756600618362427, 
    0.002292768796905875)
    
    20Hz, max=100Hz, degree=175
    (0.008750724606215954, 
    0.0005495150107890368, 
    0.02728760614991188, 
    0.0017100359546020627)
    
    20Hz, max=100Hz, degree=250
    (0.00623534107580781, 
    0.000390015309676528, 
    0.0181050356477499, 
    0.0012118576560169458) 
    
    20Hz, max=100Hz, degree=256
    (0.00615951232612133, 
    0.00038484969991259277, 
    0.017425572499632835, 
    0.0011572173098102212)
    
    20Hz, max=100Hz, degree=256, v2
    (0.006593520753085613, 
    0.0004135779454372823, 
    0.017094451934099197, 
    0.0011611274676397443)
    
    10Hz, max=100Hz, degree=256
    (0.011884157545864582, 
    0.0007408251403830945, 
    0.04682011902332306, 
    0.0026754955761134624)
    
    10Hz, max=30Hz, degree=256
    (0.004142856691032648, 
    0.00025999429635703564, 
    0.004413307178765535, 
    0.00031628430588170886)
    
    30Hz, max=100Hz, degree=256
    (0.00413433788344264, 
    0.0002601199084892869, 
    0.010248271748423576, 
    0.0007171945180743933)

    20Hz, max=100Hz, degree=33
    (0.043160390108823776, 
    0.002674056449905038, 
    0.32136210799217224, 
    0.010734910145401955)
    
    20Hz, max=100Hz, degree=20
    (0.07025660574436188, 
    0.004354883451014757, 
    0.9715675711631775, 
    0.018650120124220848)
    
    20Hz, max=50Hz, degree=20
    (0.03231345862150192, 
    0.002023769076913595, 
    0.08683804422616959, 
    0.004143711179494858)
    
    20Hz, max=50Hz, degree=20, g_li=0.003
    (0.022178268060088158, 
    0.0013867146335542202, 
    0.09227322041988373, 
    0.004128940403461456)
    '''

    @staticmethod
    def _random_initialize_for_dti_distributation_block(path, total_neurons, gui, degree, minmum_neurons_for_block=None,
                                                        dtype=['single']):
        file = loadmat('./DTI_T1_92ROI')
        block_size = file['t1_roi'][:, 0]
        block_size /= block_size.sum(0)
        dti = torch.from_numpy(np.float32(file['weight_dti']))
        dti /= dti.std(1, keepdim=True)
        dti += dti.logsumexp(1).diag()
        dti /= dti.sum(1, keepdim=True)

        if minmum_neurons_for_block is None:
            minmum_neurons_for_block = degree * 100

        kwords = [{"V_th": -50,
                   "V_reset": -65,
                   'g_Li': 0.03,
                   'g_ui': gui,
                   "E_number": int(0.8 * max(i * total_neurons, minmum_neurons_for_block)),
                   "I_number": int(0.2 * max(i * total_neurons, minmum_neurons_for_block))}
                  for i in block_size]

        return connect_for_multi_sparse_block(dti, kwords,
                                              degree=degree,
                                              init_min=0,
                                              init_max=1,
                                              perfix=path,
                                              dtype=dtype)

    @staticmethod
    def _add_laminar_cortex_model(conn_prob, gm):
        lcm_connect_prob = np.array([[907, 1600, 907, 160, 0, 0, 0, 0, 0, 0, 7752],
                                     [73, 899, 560, 151, 9, 0, 9, 0, 0, 0, 7191],
                                     [0, 133, 3557, 799, 883, 46, 431, 0, 133, 46, 1019],
                                     [0, 54, 1769, 509, 443, 28, 215, 0, 69, 23, 429],
                                     [0, 27, 416, 79, 1073, 488, 82, 0, 1684, 305, 1507],
                                     [0, 0, 168, 39, 635, 357, 35, 0, 1024, 182, 829],
                                     [0, 138, 2526, 168, 756, 71, 620, 85, 360, 547, 1510],
                                     [0, 0, 1356, 75, 382, 33, 376, 66, 128, 340, 227],
                                     [0, 2, 646, 44, 554, 111, 330, 24, 1100, 784, 2602],
                                     [0, 0, 81, 6, 93, 3, 161, 13, 464, 496, 1887]], dtype=np.float32)

        #                            synaptic layer
        # ---------------------------------------------------------------
        # Target population |       1       2/3     4       5       6
        #       2/3E        |       0.567
        #       2/3I        |               0.16
        #       4E          |       0.18    0.84    0.73
        #       4I          |                       0.16
        #       5E          |       0.25            0.02    0.76
        #       5I          |                               0.1
        #       6E          |       0.003           0.09    0.14    0.85
        #       6I          |                                       0.15

        prop = np.array([[0.567, 0., 0., 0., 0.],
                         [0., 0.16, 0., 0., 0.],
                         [0.18, 0.84, 0.73, 0., 0.],
                         [0., 0., 0.16, 0., 0.],
                         [0.25, 0., 0.02, 0.76, 0.],
                         [0., 0., 0., 0.1, 0.],
                         [0.003, 0., 0.09, 0.14, 0.85],
                         [0., 0., 0., 0., 0.15]], dtype=np.float
                        )
        cc_syn = np.matmul(prop, (lcm_connect_prob[::2, -1] + lcm_connect_prob[1::2, -1])[:, None]).squeeze()
        lcm_connect_prob[:, :2] = 0
        lcm_connect_prob[:2, :] = 0
        lcm_connect_prob[2:, -1] = cc_syn

        lcm_gm = np.array([0, 0,
                           33.8 * 78, 33.8 * 22,
                           34.9 * 80, 34.9 * 20,
                           7.6 * 82, 7.6 * 18,
                           22.1 * 83, 22.1 * 17], dtype=np.float32)  # ignore the L1 neurons
        lcm_gm /= lcm_gm.sum()

        lcm_degree_scale = lcm_connect_prob.sum(1) / lcm_connect_prob.sum() / lcm_gm
        lcm_degree_scale = np.where(np.isnan(lcm_degree_scale), 0, lcm_degree_scale)
        lcm_connect_prob /= lcm_connect_prob.sum(axis=1, keepdims=True)

        if conn_prob.shape[0] == 1:
            conn_prob[:, :] = 1
        else:
            conn_prob[np.diag_indices(conn_prob.shape[0])] = 0
            conn_prob = conn_prob / conn_prob.sum(axis=1, keepdims=True)

        out_gm = (gm[:, None] * lcm_gm[None, :]).reshape([-1])
        out_degree_scale = np.broadcast_to(lcm_degree_scale[None, :], [gm.shape[0], lcm_gm.shape[0]]).reshape([-1])
        conn_prob = sparse.COO(conn_prob)
        # only e5 is allowed to output.
        corrds1 = np.empty([4, conn_prob.coords.shape[1] * lcm_connect_prob.shape[0]], dtype=np.int64)
        corrds1[3, :] = 6
        corrds1[(0, 2), :] = np.broadcast_to(conn_prob.coords[:, :, None],
                                             [2, conn_prob.coords.shape[1], lcm_connect_prob.shape[0]]).reshape([2, -1])
        corrds1[(1), :] = np.broadcast_to(np.arange(lcm_connect_prob.shape[0], dtype=np.int64)[None, :],
                                          [conn_prob.coords.shape[1], lcm_connect_prob.shape[0]]).reshape([1, -1])

        data1 = (conn_prob.data[:, None] * lcm_connect_prob[:, -1]).reshape([-1])

        lcm_connect_prob_inner = sparse.COO(lcm_connect_prob[:, :-1])
        corrds2 = np.empty([4, conn_prob.shape[0] * lcm_connect_prob_inner.data.shape[0]], dtype=np.int64)
        corrds2[0, :] = np.broadcast_to(np.arange(conn_prob.shape[0], dtype=np.int64)[:, None],
                                        [conn_prob.shape[0], lcm_connect_prob_inner.data.shape[0]]).reshape([-1])
        corrds2[2, :] = corrds2[0, :]
        corrds2[(1, 3), :] = np.broadcast_to(lcm_connect_prob_inner.coords[:, None, :],
                                             [2, conn_prob.shape[0], lcm_connect_prob_inner.coords.shape[1]]).reshape(
            [2, -1])
        data2 = np.broadcast_to(lcm_connect_prob_inner.data[None, :],
                                [conn_prob.shape[0], lcm_connect_prob_inner.coords.shape[1]]).reshape([-1])

        out_conn_prob = sparse.COO(coords=np.concatenate([corrds1, corrds2], axis=1),
                                   data=np.concatenate([data1, data2], axis=0),
                                   shape=[conn_prob.shape[0], lcm_connect_prob.shape[0], conn_prob.shape[1],
                                          lcm_connect_prob.shape[1] - 1])

        out_conn_prob = out_conn_prob.reshape((conn_prob.shape[0] * lcm_connect_prob.shape[0],
                                               conn_prob.shape[1] * (lcm_connect_prob.shape[1] - 1)))
        if conn_prob.shape[0] == 1:
            out_conn_prob = out_conn_prob / out_conn_prob.sum(axis=1, keepdims=True)
        return out_conn_prob, out_gm, out_degree_scale

    def _random_initialize_for_lcm_block(self, scale, gui, degree, minmum_neurons_for_block=10, dtype=['single'],
                                         path=None):
        conn_prob = np.load('./conn_prob_v2.npz')['conn_prob']
        block_size = np.load('./conn_prob_v2.npz')['block_size']
        conn_prob, block_size, degree_scale = self._add_laminar_cortex_model(conn_prob, block_size)

        degree = np.maximum((degree * degree_scale).astype(np.uint16), 1)

        kwords = [{"V_th": -50,
                   "V_reset": -65,
                   'g_Li': 0.03,
                   'g_ui': gui,
                   "E_number": int(max(b * scale, minmum_neurons_for_block)) if i % 2 == 0 else 0,
                   "I_number": int(max(b * scale, minmum_neurons_for_block)) if i % 2 == 1 else 0,
                   }
                  for i, b in enumerate(block_size)]

        return connect_for_multi_sparse_block(conn_prob, kwords,
                                              degree=degree,
                                              init_min=0,
                                              init_max=1,
                                              perfix=path,
                                              dtype=dtype,
                                              split_EI=True)

    @staticmethod
    def _random_initialize_for_cortical_and_subcortical_block(scale, degree,
                                                              perfix='/public/home/ssct004t/project/zenglb/spiking_nn_for_simulation/cortical_and_subcortical',
                                                              minmum_neurons_for_block=None, dtype=['single'],
                                                              map=None):
        print('reading cortical_and_subcortical_size_info.npz')
        degree_scale = np.load(perfix + '_size_info_22703.npz')['degree']
        block_size = np.load(perfix + '_size_info_22703.npz')['size']
        name = np.load(perfix + '_size_info_22703.npz')['idx']

        print('reading cortical_and_subcortical_conn_prob.pickle')
        with open(perfix + '_conn_prob_22703.pickle', 'rb') as f:
            conn_prob = pickle.load(f)

        print('reading done')
        if minmum_neurons_for_block is None:
            minmum_neurons_for_block = degree * 100

        degree = np.maximum((degree * degree_scale.astype(np.float) / 100).astype(np.uint16), 1)

        if map is None:
            map = './map_10000_v4_cortical_v2.pkl'

        with open(map, 'rb') as f:
            merge_map = pickle.load(f)
        assert isinstance(merge_map, dict)

        max_block_size = max([np.sum(block_size[v] * degree[v]) for v in merge_map.values()])
        total_block_size = sum([np.sum(block_size[v] * degree[v]) for v in merge_map.values()])
        scale /= max_block_size / (total_block_size / sum([np.sum(block_size[v]) for v in merge_map.values()]))

        '''
        for i in range(3):
            memory_scale = './memory_record_10000_{}.npy'.format(i+1)
            if os.path.exists(memory_scale):
                print('load', memory_scale)
                memory_scale = np.load().astype(np.float)
                memory_scale = memory_scale[:, 1] - memory_scale[:, 0]
                memory_scale /= memory_scale.mean()
                block_size = block_size.astype(np.float) 
                for i in range(len(merge_map)):
                    block_size[np.array(merge_map[str(i)])]/= memory_scale[i]
        '''

        print('estimate neurons: {}'.format(sum([int(max(b * scale, minmum_neurons_for_block)) for b in block_size])))
        order = np.concatenate([np.array(merge_map[str(i)], dtype=np.int64) for i in range(len(merge_map))])
        assert np.unique(order).shape[0] == order.shape[0]

        block_size = np.ascontiguousarray(block_size[order])
        degree = np.ascontiguousarray(degree[order])

        if isinstance(conn_prob, np.ndarray):
            conn_prob = np.ascontiguousarray(conn_prob[:, order])
        else:
            argsort_order = np.argsort(order)
            nonzeros = np.logical_and(np.isfinite(conn_prob.data), conn_prob.data > 0).nonzero()[0]

            def new_coord(coord):
                sorted_coord, idx = np.unique(coord[nonzeros], return_inverse=True)
                assert np.all(sorted_coord == order[argsort_order]), "1"
                idx = argsort_order[idx]
                assert np.all(order[idx] == coord[nonzeros]), "2"
                return idx

            conn_prob = sparse.COO(coords=np.stack([conn_prob.coords[0][nonzeros],
                                                    new_coord(conn_prob.coords[1])]),
                                   data=conn_prob.data[nonzeros],
                                   shape=[len(block_size), len(block_size)])
        name = name[order]

        def gui(i):
            # for 100 degree
            gui_laminar = np.array([[0.00659512, 0.00093751, 0.1019024, 0.00458985],
                                    [0.01381911, 0.00196363, 0.18183651, 0.00727698],
                                    [0.00659512, 0.00093751, 0.1019024, 0.00458985],
                                    [0.01381911, 0.00196363, 0.18183651, 0.00727698],
                                    [0.00754673, 0.00106148, 0.09852575, 0.00431849],
                                    [0.0134587, 0.00189199, 0.15924831, 0.00651926],
                                    [0.00643689, 0.00091055, 0.10209763, 0.00444712],
                                    [0.01647443, 0.00234132, 0.21505809, 0.00796669],
                                    [0.00680198, 0.00095797, 0.06918744, 0.00324963],
                                    [0.01438906, 0.00202573, 0.14674303, 0.00587307]], dtype=np.float64)
            gui_voxel = np.array([0.00618016, 0.00086915, 0.05027743, 0.00253291], dtype=np.float64)
            '''
            voxel_idx = i // 10
            layer_idx = i % 10
            if voxel_idx * 10 + 2 in order:
                return tuple(gui_laminar[layer_idx].tolist())
            else:
                return tuple(gui_voxel.tolist())
            '''
            return tuple(gui_voxel.tolist())

        kwords = [{"V_th": -50,
                   "V_reset": -65,
                   'g_Li': 0.03,
                   'g_ui': gui(i),
                   "E_number": int(max(b * scale, minmum_neurons_for_block)) if i % 2 == 0 else 0,
                   "I_number": int(max(b * scale, minmum_neurons_for_block)) if i % 2 == 1 else 0,
                   'sub_block_idx': n,
                   }
                  for i, n, b in zip(order, name, block_size)]

        partation = [len(merge_map[str(i)]) for i in range(len(merge_map))]
        return connect_for_multi_sparse_block(conn_prob, kwords,
                                              degree=degree,
                                              init_min=0,
                                              init_max=1,
                                              perfix=None,
                                              dtype=dtype,
                                              name=name,
                                              split_EI=True), partation

    @staticmethod
    def _random_initialize_for_cortical_and_subcortical_block_lyh(scale, gui, degree, map_path,
                                                                  minmum_neurons_for_block=None, dtype=['single']):
        # print('reading cortical_and_subcortical_size_info.npz')
        conn_root = '/public/home/ssct005t/lyh_route/tables/conn_table/cortical_v2/'

        size_name = 'cortical_and_subcortical_size_info_22703.npz'
        conn_name = 'cortical_and_subcortical_conn_prob_22703.pickle'

        degree_scale = np.load(conn_root + size_name)['degree']
        block_size = np.load(conn_root + size_name)['size']

        # print('reading cortical_and_subcortical_conn_prob.pickle')
        with open(conn_root + conn_name, 'rb') as f:
            conn_prob = pickle.load(f)

        # print('reading done')
        if minmum_neurons_for_block is None:
            minmum_neurons_for_block = degree * 100

        degree = np.maximum((degree * degree_scale.astype(np.float) / 100).astype(np.uint16), 1)

        with open(map_path, 'rb') as f:
            merge_map = pickle.load(f)
        assert isinstance(merge_map, dict)

        max_block_size = max([np.sum(block_size[v] * degree[v]) for v in merge_map.values()])
        total_block_size = sum([np.sum(block_size[v] * degree[v]) for v in merge_map.values()])
        scale /= max_block_size / (total_block_size / sum([np.sum(block_size[v]) for v in merge_map.values()]))

        memory_scale = './memory_record_10000.npy'
        if os.path.exists(memory_scale):
            print('load memory record')
            memory_scale = np.load(memory_scale).astype(np.float)
            memory_scale = memory_scale[:, 1] - memory_scale[:, 0]
            memory_scale /= memory_scale.mean()
            block_size = block_size.astype(np.float)
            for i in range(len(merge_map)):
                block_size[np.array(merge_map[str(i)])] /= memory_scale[i]

        print('estimate neurons: {}'.format(sum([int(max(b * scale, minmum_neurons_for_block)) for b in block_size])))
        order = np.concatenate([np.array(merge_map[str(i)], dtype=np.int64) for i in range(len(merge_map))])
        assert np.unique(order).shape[0] == order.shape[0]

        block_size = np.ascontiguousarray(block_size[order])
        degree = np.ascontiguousarray(degree[order])

        if isinstance(conn_prob, np.ndarray):
            conn_prob = np.ascontiguousarray(conn_prob[order, :][:, order])
        else:
            argsort_order = np.argsort(order)
            nonzeros = np.logical_and(np.isfinite(conn_prob.data), conn_prob.data > 0).nonzero()[0]

            def new_coord(coord):
                sorted_coord, idx = np.unique(coord[nonzeros], return_inverse=True)
                assert np.all(sorted_coord == order[argsort_order]), "1"
                idx = argsort_order[idx]
                assert np.all(order[idx] == coord[nonzeros]), "2"
                return idx

            conn_prob = sparse.COO(coords=np.stack([new_coord(conn_prob.coords[0]),
                                                    new_coord(conn_prob.coords[1])]),
                                   data=conn_prob.data[nonzeros],
                                   shape=[len(block_size), len(block_size)])

        kwords = [{"V_th": -50,
                   "V_reset": -65,
                   'g_Li': 0.03,
                   'g_ui': gui,
                   "E_number": int(max(b * scale, minmum_neurons_for_block)) if i % 2 == 0 else 0,
                   "I_number": int(max(b * scale, minmum_neurons_for_block)) if i % 2 == 1 else 0,
                   'sub_block_idx': i
                   }
                  for i, b in zip(order, block_size)]

        partation = [len(merge_map[str(i)]) for i in range(len(merge_map))]
        return connect_for_multi_sparse_block(conn_prob, kwords,
                                              degree=degree,
                                              init_min=0,
                                              init_max=1,
                                              perfix=None,
                                              dtype=dtype,
                                              split_EI=True), partation

    def _random_initialize_for_voxel_block(scale, gui, degree, minmum_neurons_for_block=None, dtype=['single']):
        conn_prob = np.load('./conn_prob_v2.npz')['conn_prob']
        block_size = np.load('conn_prob_v2.npz')['block_size']

        if minmum_neurons_for_block is None:
            minmum_neurons_for_block = degree * 100

        with open('map_1200_v3.pkl', 'rb') as f:
            merge_map = pickle.load(f)
        assert isinstance(merge_map, dict)

        max_block_size = max([np.sum(block_size[v]) for v in merge_map.values()])
        scale /= max_block_size
        print('estimate neurons: {}'.format(int(scale)))

        order = np.concatenate([np.array(merge_map[str(i)], dtype=np.int64) for i in range(len(merge_map))])
        assert np.unique(order).shape[0] == order.shape[0]

        conn_prob = np.ascontiguousarray(conn_prob[order, :][:, order])
        block_size = np.ascontiguousarray(block_size[order])

        kwords = [{"V_th": -50,
                   "V_reset": -65,
                   'g_Li': 0.03,
                   'g_ui': gui,
                   "E_number": int(0.8 * max(b * scale, minmum_neurons_for_block)),
                   "I_number": int(0.2 * max(b * scale, minmum_neurons_for_block)),
                   'sub_block_idx': i
                   }
                  for i, b in zip(order, block_size)]

        partation = [len(merge_map[str(i)]) for i in range(len(merge_map))]
        return connect_for_multi_sparse_block(conn_prob, kwords,
                                              degree=degree,
                                              init_min=0,
                                              init_max=1,
                                              perfix=None,
                                              dtype=dtype), partation

    @staticmethod
    def _random_initialize_for_voxel_block_lyh(scale, gui, degree, map_path, minmum_neurons_for_block=None,
                                               dtype=['single']):
        conn_root = "/public/home/ssct005t/lyh_route/voxel_22703/"
        conn_prob = np.load(conn_root + 'conn.npy')
        block_size = np.load(conn_root + 'size.npy')

        if minmum_neurons_for_block is None:
            minmum_neurons_for_block = degree * 100

        with open(map_path, 'rb') as f:
            merge_map = pickle.load(f)
        assert isinstance(merge_map, dict)
        print(map_path + 'loaded.')

        max_block_size = max([np.sum(block_size[v]) for v in merge_map.values()])
        scale /= max_block_size
        print('estimate neurons: {}'.format(int(scale)))

        order = np.concatenate([np.array(merge_map[str(i)], dtype=np.int64) for i in range(len(merge_map))])
        assert np.unique(order).shape[0] == order.shape[0]

        conn_prob = np.ascontiguousarray(conn_prob[order, :][:, order])
        block_size = np.ascontiguousarray(block_size[order])

        kwords = [{"V_th": -50,
                   "V_reset": -65,
                   'g_Li': 0.03,
                   'g_ui': gui,
                   "E_number": int(0.8 * max(b * scale, minmum_neurons_for_block)),
                   "I_number": int(0.2 * max(b * scale, minmum_neurons_for_block)),
                   'sub_block_idx': i
                   }
                  for i, b in zip(order, block_size)]

        partation = [len(merge_map[str(i)]) for i in range(len(merge_map))]
        return connect_for_multi_sparse_block(conn_prob, kwords,
                                              degree=degree,
                                              init_min=0,
                                              init_max=1,
                                              perfix=None,
                                              dtype=dtype), partation

    @staticmethod
    def _find_gui_in_1000_block(delta_t=1, default_Hz=30, max_output_Hz=100, T_ref=5, degree=256, g_Li=0.03, V_L=-75,
                                V_rst=-65, V_th=-50, path="./single_small_test/", need_test=False):
        prob = torch.tensor([[1.]])

        gap = V_th - V_rst

        noise_rate = 1 / (1000 / delta_t / default_Hz)
        max_delta_raise = gap / (1000 / delta_t / max_output_Hz - T_ref)
        default_delta_raise = gap / (1000 / delta_t / default_Hz - T_ref)

        leaky_compensation = g_Li * ((V_th + V_rst) / 2 - V_L)

        label = torch.tensor([0.3 * (max_delta_raise + leaky_compensation),
                              0.7 * (max_delta_raise + leaky_compensation),
                              0.5 * (max_delta_raise - default_delta_raise),
                              0.5 * (max_delta_raise - default_delta_raise)])
        print(label.tolist())

        gui = label

        def test_gui(max_iter=4000, noise_rate=noise_rate):
            property, w_uij = connect_for_block(os.path.join(path, 'single'))

            B = block(
                node_property=property,
                w_uij=w_uij,
                delta_t=delta_t,
            )
            out_list = []
            Hz_list = []

            for k in range(max_iter):
                B.run(noise_rate=noise_rate, isolated=True)
                out_list.append(B.I_ui.mean(-1).abs())
                Hz_list.append(float(B.active.sum()) / property.shape[0])
            out = torch.stack(out_list[-500:]).mean(0)
            Hz = sum(Hz_list[-500:]) * 1000 / 500
            print('out:', out.tolist(), Hz)
            return out

        for i in range(20):
            connect_for_multi_sparse_block(prob, {'g_Li': g_Li,
                                                  'g_ui': gui,
                                                  "V_reset": -65},
                                           E_number=int(1.6e3), I_number=int(4e2), degree=degree, init_min=0,
                                           init_max=1, perfix=path)
            gui = gui * label / test_gui()
            print(i)
            print('gui:', gui.tolist())

        if need_test:
            connect_for_multi_sparse_block(prob, {'g_Li': g_Li,
                                                  'g_ui': gui,
                                                  "V_reset": -65},
                                           E_number=int(1.6e3), I_number=int(4e2), degree=degree, init_min=0,
                                           init_max=1, perfix=path)
            for i in range(0, 200, 5):
                print("testing ", i)
                test_gui(noise_rate=i / 1000)

        print(gui.shape)

        return tuple(gui.tolist())

    def _test_random_initialize_for_single_small_block(self):
        prob = torch.tensor([[1.]])
        connect_for_multi_sparse_block(prob, {'g_Li': 0.003,
                                              'g_ui': (0.022178268060088158,
                                                       0.0013867146335542202,
                                                       0.09227322041988373,
                                                       0.004128940403461456),
                                              "V_reset": -65,
                                              "V_th": -50},
                                       E_number=int(8e2), I_number=int(2e2), degree=int(20), init_min=0, init_max=1,
                                       perfix="./single_small/")

    def _test_random_initialize_for_10k_with_multi_degree(self):
        degree_list = [5, 10, 20, 50, 100]
        prob = torch.tensor([[1.]])
        g_Li = 0.03
        path = './single_10k'
        os.makedirs(path, exist_ok=True)

        for d in degree_list:
            print('processing', d)
            gui = self._find_gui_in_1000_block(degree=d, g_Li=g_Li)
            old_form_dir = os.path.join(path, 'degree_{}'.format(d))
            connect_for_multi_sparse_block(prob, {'g_Li': g_Li,
                                                  'g_ui': gui,
                                                  "V_reset": -65,
                                                  "V_th": -50},
                                           E_number=int(8e3), I_number=int(2e3), degree=int(d), init_min=0, init_max=1,
                                           perfix=old_form_dir)

    def _test_random_initialize_for_dti_distributation_block_200k(self):
        file = loadmat('./GM_AAL_age50')
        block_size = file['GM_AAL_age50'].sum(0)[:90]
        block_size /= block_size.sum(0)
        file = loadmat('./matrix_hcp')
        dti = torch.from_numpy(np.float32(file['matrix_HCP'])).mean(0)[:90, :90]
        merge_group = [(4, 8),
                       (10, 12),
                       (16, 18),
                       (20, 30),
                       (34, 36),
                       (38, 40),
                       (24, 26),
                       (42, 44),
                       (48, 52),
                       (62, 64),
                       (68, 70, 72),
                       (74, 76),
                       (78, 80),
                       (82, 86)]

        delete_list = set()

        for group in merge_group:
            suffix = [0, 1]
            for s in suffix:
                base = group[0] + s
                for _idx in group[1:]:
                    idx = _idx + s
                    delete_list.add(idx)
                    dti[base, :] += dti[idx, :]
                    dti[:, base] += dti[:, idx]
                    block_size[base] += block_size[idx]

        exist_list = [i for i in range(90) if i not in delete_list]
        dti[exist_list, exist_list] = 0
        dti = dti[exist_list, :][:, exist_list]
        block_size = block_size[exist_list]

        dti /= dti.std(1, keepdim=True)
        dti += dti.logsumexp(1).diag()
        dti /= dti.sum(1, keepdim=True)

        total_neurons = int(2e5)

        kwords = [{"V_th": -50,
                   "V_reset": -65,
                   'g_Li': 0.003,
                   'g_ui': (0.022178268060088158,
                            0.0013867146335542202,
                            0.09227322041988373,
                            0.004128940403461456),
                   "E_number": int(0.8 * max(i * total_neurons, 2000)),
                   "I_number": int(0.2 * max(i * total_neurons, 2000))}
                  for i in block_size]

        conn = connect_for_multi_sparse_block(dti, kwords,
                                              degree=int(20),
                                              init_min=0,
                                              init_max=1,
                                              perfix=None,
                                              dtype=["single"])

        out_path = "dti_single_200k_new"
        os.makedirs(out_path, exist_ok=True)
        merge_dti_distributation_block(conn,
                                       out_path,
                                       dtype=["single"],
                                       number=1,
                                       debug_block_path="./single_small/single")

    def _test_random_initialize_for_dti_single_block(self):
        path = "./dti_single_500k/"
        os.makedirs(path, exist_ok=True)
        self._random_initialize_for_dti_distributation_block('./dti_distribution_500k', int(5e5), (0.001617693924345076,
                                                                                                   0.0001014432345982641,
                                                                                                   0.003837576135993004,
                                                                                                   0.000286647496977821),
                                                             1000)
        merge_dti_distributation_block("./dti_distribution_500k/single", path, dtype=["single", "half"])

    def _test_random_initialize_for_dti_single_block_50m(self):
        path = "./dti_24_50m/"
        os.makedirs(path, exist_ok=True)

        self._random_initialize_for_dti_distributation_block('./dti_distribution_50m', int(5e7), (0.00615951232612133,
                                                                                                  0.00038484969991259277,
                                                                                                  0.017425572499632835,
                                                                                                  0.0011572173098102212),
                                                             256)

        block_threshhold = merge_dti_distributation_block("./dti_distribution_50m/single",
                                                          path,
                                                          number=24,
                                                          dtype=["single"],
                                                          debug_block_path="./single_small/single")
        size = block_threshhold[-1]

        sample_rate = 0.02

        debug_selection_idx = np.load(os.path.join(path, "debug_selection_idx.npy"))
        debug_selection_idx = block_threshhold[debug_selection_idx[:, 0]] + debug_selection_idx[:, 1]

        sample_selection_idx = np.random.choice(size, int(sample_rate * 2 * size), replace=False)
        sample_selection_idx = np.array(list(set(sample_selection_idx) - set(debug_selection_idx)))
        sample_selection_idx = np.random.permutation(sample_selection_idx)[:int(sample_rate * size)]

        assert sample_selection_idx.shape[0] == int(sample_rate * size)

        sample_block_idx, sample_neuron_idx = turn_to_block_idx(sample_selection_idx, block_threshhold)

        np.save(os.path.join(path, "sample_selection_idx"),
                np.ascontiguousarray(
                    np.stack([sample_block_idx, sample_neuron_idx], axis=1)))

    def _test_random_initialize_for_dti_16_maximum(self):
        for num in range(5000000, 5500000, 500000):
            path = "./dti_16_{}_noise_rate_free/".format(num)
            total_blocks = 16
            os.makedirs(path, exist_ok=True)

            # gui = self._find_gui_in_1000_block(degree=100)
            gui = tuple(np.load(
                '/public/home/ssct004t/project/spiking_nn_for_brain_simulation/dti_2k_10G_2/single/block_0.npz')[
                            'property'][0, 10:14].tolist())
            conn = self._random_initialize_for_dti_distributation_block(None, 16 * num, gui, 100)

            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()

            for i in range(rank, total_blocks, size):
                merge_dti_distributation_block(conn, path,
                                               MPI_rank=i,
                                               number=16,
                                               dtype=["single"],
                                               debug_block_path="./single_small/single",
                                               only_load=(i != 0))

    def _test_random_initialize_for_dti_10k_86G(self):
        path = "/public/home/ssct004t/project/spiking_nn_for_brain_simulation/dti_10k_86G"
        total_blocks = 10000
        os.makedirs(path, exist_ok=True)

        # gui = self._find_gui_in_1000_block(degree=100)
        conn, partation = self._random_initialize_for_cortical_and_subcortical_block(4000000, 100)

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        for i in range(rank, total_blocks, size):
            merge_dti_distributation_block(conn, path,
                                           MPI_rank=i,
                                           block_partation=partation,
                                           dtype=["single"],
                                           debug_block_path=None,  # "./dti_single_200k_new/single",
                                           only_load=(i != 0))

    def _test_random_initialize_for_5G_to_10G(self):
        paths = "/public/home/ssct004t/project/spiking_nn_for_brain_simulation/dti_comp_voxel"
        os.makedirs(paths, exist_ok=True)

        names = ["dti_1k_5G",
                 "dti_1_2k_6G",
                 "dti_1_4k_7G",
                 "dti_1_6k_8G",
                 "dti_1_8k_9G",
                 "dti_2k_10G"]

        maps = ['/public/home/ssct005t/project/wml_istbi/tables/map_table/map_1000_v1_cortical_v2.pkl',
                '/public/home/ssct005t/project/wml_istbi/tables/map_table/map_1200_v1_cortical_v2.pkl',
                '/public/home/ssct005t/project/wml_istbi/tables/map_table/map_1400_v1_cortical_v2.pkl',
                '/public/home/ssct005t/project/wml_istbi/tables/map_table/map_1600_v1_cortical_v2.pkl',
                '/public/home/ssct005t/project/wml_istbi/tables/map_table/map_1800_v1_cortical_v2.pkl',
                '/public/home/ssct005t/project/wml_istbi/tables/map_table/map_2000_v1_cortical_v2.pkl']

        total_blocks = [1000, 1200, 1400, 1600, 1800, 2000]

        # gui = self._find_gui_in_1000_block(degree=100)
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        i = rank

        for name, map, total_block in zip(names, maps, total_blocks):
            conn, partation = self._random_initialize_for_cortical_and_subcortical_block(5000000, 100, map=map,
                                                                                         perfix='voxel')
            path = os.path.join(paths, name)
            os.makedirs(path, exist_ok=True)

            while i < total_block:
                merge_dti_distributation_block(conn, path,
                                               MPI_rank=i,
                                               block_partation=partation,
                                               dtype=["single"],
                                               debug_block_path=None,  # "./dti_single_200k_new/single",
                                               only_load=(i != 0))
                i += size

            i -= total_block

    def _test_random_initialize_for_5G_to_10G_2(self):
        paths = "/public/home/ssct004t/project/spiking_nn_for_brain_simulation/dti_comp_aal1"
        os.makedirs(paths, exist_ok=True)

        names = ["dti_1k_5G",
                 "dti_1_2k_6G",
                 "dti_1_4k_7G",
                 "dti_1_6k_8G",
                 "dti_1_8k_9G",
                 "dti_2k_10G"]

        maps = ['/public/home/ssct005t/project/wml_istbi/tables/map_table/map_1000_v1_cortical_v2.pkl',
                '/public/home/ssct005t/project/wml_istbi/tables/map_table/map_1200_v1_cortical_v2.pkl',
                '/public/home/ssct005t/project/wml_istbi/tables/map_table/map_1400_v1_cortical_v2.pkl',
                '/public/home/ssct005t/project/wml_istbi/tables/map_table/map_1600_v1_cortical_v2.pkl',
                '/public/home/ssct005t/project/wml_istbi/tables/map_table/map_1800_v1_cortical_v2.pkl',
                '/public/home/ssct005t/project/wml_istbi/tables/map_table/map_2000_v1_cortical_v2.pkl']

        total_blocks = [1000, 1200, 1400, 1600, 1800, 2000]

        # gui = self._find_gui_in_1000_block(degree=100)
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        i = rank

        for name, map, total_block in zip(names, maps, total_blocks):
            conn, partation = self._random_initialize_for_cortical_and_subcortical_block(5000000, 100, map=map,
                                                                                         perfix='aal1')
            path = os.path.join(paths, name)
            os.makedirs(path, exist_ok=True)

            while i < total_block:
                merge_dti_distributation_block(conn, path,
                                               MPI_rank=i,
                                               block_partation=partation,
                                               dtype=["single"],
                                               debug_block_path=None,  # "./dti_single_200k_new/single",
                                               only_load=(i != 0))
                i += size

            i -= total_block

    def test_random_initialize_for_5G_to_10G_2_lyh(self):
        conn_version = "cortical_v2"
        k_neurons_per_gpu = int(5e6)  # 8e6 is the upper bound

        # dti data folder name
        names = ["dti_6G_v12"]

        # map path
        maps = ["/public/home/ssct005t/lyh_route/tables/map_table/map_1200_v12_cortical_v2.pkl"]

        # GPU number used
        total_blocks = [1200]

        assert len(names) == len(maps) == len(total_blocks)

        paths = "/public/home/ssct004t/project/1201/" + conn_version + "_experiments"
        os.makedirs(paths, exist_ok=True)

        # gui = self._find_gui_in_1000_block(degree=100)
        gui = tuple(np.load(
            '/public/home/ssct004t/project/spiking_nn_for_brain_simulation/dti_2k_10G_2/single/block_0.npz')[
                        'property'][0, 10:14].tolist())

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        i = rank

        for name, map, total_block in zip(names, maps, total_blocks):
            conn, partation = self._random_initialize_for_cortical_and_subcortical_block_lyh(scale=k_neurons_per_gpu,
                                                                                             degree=100, gui=gui,
                                                                                             map_path=map)
            path = os.path.join(paths, name)
            os.makedirs(path, exist_ok=True)

            while i < total_block:
                merge_dti_distributation_block(conn, path,
                                               MPI_rank=i,
                                               block_partation=partation,
                                               dtype=["single"],
                                               debug_block_path=None,  # "./dti_single_200k_new/single",
                                               only_load=(i != 0))
                i += size

            i -= total_block

    def _test_random_initialize_for_dti_voxel_lyh(self):
        total_blocks = int(2000)
        conn_version = "voxel_22703"
        map_version = "map_" + str(total_blocks) + "_v1_" + conn_version
        k_neurons_per_gpu = int(5e6)  # 8e6 is the upper bound

        map_path = "/public/home/ssct005t/lyh_route/tables/map_table/" + map_version + ".pkl"

        path = "/public/home/ssct005t/project/1201/dti_" + "map_version"
        os.makedirs(path, exist_ok=True)

        # gui = self._find_gui_in_1000_block(degree=100)
        gui = tuple(np.load(
            '/public/home/ssct004t/project/spiking_nn_for_brain_simulation/dti_2k_10G_2/single/block_0.npz')[
                        'property'][0, 10:14].tolist())

        conn, partation = self._random_initialize_for_voxel_block(scale=k_neurons_per_gpu, gui=gui, degree=100,
                                                                  map_path=map_path)

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if rank == 0:
            print("#" * 41)
            print("Total blocks:", total_blocks)
            print("Conn version:", conn_version)
            print("Map version:", map_version)
            print("K neurons per GPU:", k_neurons_per_gpu)
            print("#" * 41)
            print(gui)

        debug_block_path = "/public/home/ssct004t/project/spiking_nn_for_brain_simulation/dti_single_200k_new/single"
        for i in range(rank, total_blocks, size):
            merge_dti_distributation_block(conn, path,
                                           MPI_rank=i,
                                           block_partation=partation,
                                           dtype=["single"],
                                           debug_block_path=debug_block_path,
                                           only_load=(i != 0))

    def _test_random_initialize_for_dti_1200_6G_lyh_old(self):
        path = "./lyh_dt/i_1200_6G_v2/"
        total_blocks = 1200
        os.makedirs(path, exist_ok=True)

        # gui = self._find_gui_in_1000_block(degree=100)
        gui = tuple(np.load(
            '/public/home/ssct004t/project/spiking_nn_for_brain_simulation/dti_2k_10G_2/single/block_0.npz')[
                        'property'][0, 10:14].tolist())

        conn, partation = self._random_initialize_for_voxel_block(5000000, gui, 100)

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if rank == 0:
            print(gui)

        for i in range(rank, total_blocks, size):
            merge_dti_distributation_block(conn, path,
                                           MPI_rank=i,
                                           block_partation=partation,
                                           dtype=["single"],
                                           debug_block_path="./dti_single_200k_new/single",
                                           only_load=(i != 0))

    def _test_random_initialize_for_dti_lcm_10m_multi(self):
        degree = [100]
        block_num = [1, 2, 4]

        path = './dti_new_10m_lcm'
        os.makedirs(path, exist_ok=True)

        for d in degree:
            gui = self._find_gui_in_1000_block(degree=d)
            dti_distribution_path = os.path.join(path, 'dti_distribution_d{}'.format(d))
            self._random_initialize_for_lcm_block(int(1e7), gui, d, minmum_neurons_for_block=10,
                                                  path=dti_distribution_path)
            for n in block_num:
                dti_block_path = os.path.join(path, 'dti_n{}_d{}'.format(n, d))
                os.makedirs(dti_block_path, exist_ok=True)
                merge_dti_distributation_block(os.path.join(dti_distribution_path, 'single'),
                                               dti_block_path,
                                               dtype=["single"],
                                               number=n,
                                               debug_block_path="./single_small/single")

    def _test_random_initialize_for_dti_60_800m(self):
        path = "./dti_60_800m/"
        blocks_number = [2] * 30 + [2 / 3] * 30
        os.makedirs(path, exist_ok=True)

        gui = self._find_gui_in_1000_block(degree=100)
        # gui = tuple(np.load('./dti_distribution_600m/single/block_75.npz')['property'][0, 10:14].tolist())
        conn = self._random_initialize_for_dti_distributation_block(None, int(8e8), gui, 100)

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        for i in range(rank, len(blocks_number), size):
            merge_dti_distributation_block(conn, path,
                                           MPI_rank=i,
                                           number=blocks_number,
                                           dtype=["single"],
                                           debug_block_path="./single_small/single",
                                           only_load=(i != 0))

    def _test_random_initialize_for_dti_single_block_5m(self):
        path = "./dti_single_5m/"
        os.makedirs(path, exist_ok=True)
        self._random_initialize_for_dti_distributation_block("./dti_distribution_5m", int(5e6), (0.010177472606301308,
                                                                                                 0.0006282327813096344,
                                                                                                 0.03813745081424713,
                                                                                                 0.0021428221371024847),
                                                             100)
        merge_dti_distributation_block("./dti_distribution_5m/single", path, dtype=["single", "half"])

    def _test_random_initialize_for_dti_10m_multi(self):
        degree = [10, 20, 50, 100]
        block_num = [1, 3]

        path = './dti_new_10m'
        os.makedirs(path, exist_ok=True)

        for d in degree:
            gui = self._find_gui_in_1000_block(degree=d)
            # gui = tuple(np.load('/home1/bychen/spliking_nn_for_brain_simulation/dti_new_10m/dti_distribution_d100/single/block_0.npz')['property'][0, 10:14].tolist())
            dti_distribution_path = os.path.join(path, 'dti_distribution_d{}'.format(d))
            self._random_initialize_for_dti_distributation_block(dti_distribution_path, int(1e7), gui, d)
            for n in block_num:
                dti_block_path = os.path.join(path, 'dti_n{}_d{}'.format(n, d))
                os.makedirs(dti_block_path, exist_ok=True)
                merge_dti_distributation_block(os.path.join(dti_distribution_path, 'single'),
                                               dti_block_path,
                                               dtype=["single"],
                                               number=n,
                                               debug_block_path="./single_small/single")

    def _test_random_initialize_for_dti_single_block_200k(self):
        path = "./dti_single_200k/"
        os.makedirs(path, exist_ok=True)
        merge_dti_distributation_block("./dti_distribution_200k/single", path, dtype=["single"])

    def _test_random_initialize_for_dti_single_block_1_6m_and_half(self):
        double_path = "./dti_double_1_6m/"
        os.makedirs(double_path, exist_ok=True)

        self._random_initialize_for_dti_distributation_block('./dti_distribution_1_6m', int(1.6e6),
                                                             (0.043160390108823776,
                                                              0.002674056449905038,
                                                              0.32136210799217224,
                                                              0.010734910145401955),
                                                             33)
        merge_dti_distributation_block("./dti_distribution_1_6m/single", double_path, dtype=["single"], number=2)

        single_path = "./dti_single_1_6m/"
        os.makedirs(single_path, exist_ok=True)
        merge_dti_distributation_block("./dti_distribution_1_6m/single", single_path, dtype=["single"], number=1)

        recover_idx_name = "recover_idx.npy"
        single_recover_idx = np.load(os.path.join(single_path, recover_idx_name))
        double_recover_idx = np.load(os.path.join(double_path, recover_idx_name))

        np.save(os.path.join(double_path, "resort_idx"), np.argsort(single_recover_idx)[double_recover_idx])

        data = np.load("dti_single_1_6m/single/block_0.npz")
        single_property = data["property"]
        data = np.load("dti_double_1_6m/single/block_0.npz")
        double_A_property = data["property"]
        data = np.load("dti_double_1_6m/single/block_1.npz")
        double_B_property = data["property"]
        resort_idx = np.load("dti_double_1_6m/resort_idx.npy")

        single_noise = np.random.rand(500, single_property.shape[0]).astype(np.float32)
        np.save("dti_single_1_6m/single/sample.npy", np.ascontiguousarray(single_noise))
        np.save("dti_double_1_6m/single/sample_1.npy",
                np.ascontiguousarray(single_noise[:, resort_idx][:, double_B_property.shape[0]:]))
        np.save("dti_double_1_6m/single/sample_0.npy",
                np.ascontiguousarray(single_noise[:, resort_idx][:, :double_B_property.shape[0]]))

    def _test_random_initialize_for_dti_with_4block(self):
        prob = torch.ones([90, 90]) / 90

        kwords = [{"V_th": -50,
                   "V_reset": -65,
                   'g_Li': 0.03,
                   'g_ui': (0.07025660574436188,
                            0.004354883451014757,
                            0.9715675711631775,
                            0.018650120124220848),
                   "E_number": int(0.8 * max(4000 / 90, 0)),
                   "I_number": int(0.2 * max(4000 / 90, 0))}] * 90

        connect_for_multi_sparse_block(prob, kwords,
                                       degree=20,
                                       init_min=0,
                                       init_max=1,
                                       perfix="./dti_distribution_4k/",
                                       dtype=["single"])

        double_path = "./dti_4_4k/"
        os.makedirs(double_path, exist_ok=True)
        merge_dti_distributation_block("./dti_distribution_4k/single", double_path, dtype=["single"], number=4)

        single_path = "./dti_single_4k/"
        os.makedirs(single_path, exist_ok=True)
        merge_dti_distributation_block("./dti_distribution_4k/single", single_path, dtype=["single"], number=1)

        recover_idx_name = "recover_idx.npy"
        single_recover_idx = np.load(os.path.join(single_path, recover_idx_name))
        double_recover_idx = np.load(os.path.join(double_path, recover_idx_name))

        resort_idx = np.argsort(single_recover_idx)[double_recover_idx]
        np.save(os.path.join(double_path, "resort_idx"), np.ascontiguousarray(resort_idx))

        data = np.load(os.path.join(single_path, "single/block_0.npz"))
        single_property = data["property"]

        single_noise = np.random.rand(500, single_property.shape[0]).astype(np.float32)
        np.save(os.path.join(single_path, "single/sample.npy"), np.ascontiguousarray(single_noise))

        single_noise_after_resort = single_noise[:, resort_idx]

        base = 0
        for i in range(4):
            data = np.load(os.path.join(double_path, "single/block_{}.npz".format(i)))
            property = data["property"]
            np.save(os.path.join(double_path, "single/sample_{}.npy".format(i)),
                    np.ascontiguousarray(single_noise_after_resort[:, base:base + property.shape[0]]))
            base += property.shape[0]

        assert base == single_property.shape[0]

    def _test_random_initialize_for_dti_with_4_small_block(self):
        prob = torch.ones([90, 90]) / 90

        kwords = [{"V_th": -50,
                   "V_reset": -65,
                   'g_Li': 0.003,
                   'g_ui': (0.022178268060088158,
                            0.0013867146335542202,
                            0.09227322041988373,
                            0.004128940403461456),
                   "E_number": int(0.8 * max(4e4 / 90, 0)),
                   "I_number": int(0.2 * max(4e4 / 90, 0))}] * 90

        out = connect_for_multi_sparse_block(prob, kwords,
                                             degree=20,
                                             init_min=0,
                                             init_max=1,
                                             perfix=None,
                                             dtype=["single"])

        double_path = "./dti_4_4k/"
        os.makedirs(double_path, exist_ok=True)

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        for i in range(rank, 4, size):
            merge_dti_distributation_block(out,
                                           double_path,
                                           dtype=["single"],
                                           number=4,
                                           output_degree=False,
                                           MPI_rank=i)

    def _test_random_initialize_for_dti_with_4_big_block(self):
        prob = torch.ones([90, 90]) / 90

        kwords = [{"V_th": -50,
                   "V_reset": -65,
                   'g_Li': 0.03,
                   'g_ui': (0.001617693924345076,
                            0.0001014432345982641,
                            0.003837576135993004,
                            0.000286647496977821),
                   "E_number": int(0.8 * max(2e6 / 90, 0)),
                   "I_number": int(0.2 * max(2e6 / 90, 0))}] * 90

        out = connect_for_multi_sparse_block(prob, kwords,
                                             degree=1000,
                                             init_min=0,
                                             init_max=1,
                                             perfix=None,
                                             dtype=["single"])

        double_path = "./dti_4_2m/"
        os.makedirs(double_path, exist_ok=True)

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        for i in range(rank, 4, size):
            merge_dti_distributation_block(out,
                                           double_path,
                                           dtype=["single"],
                                           number=4,
                                           debug_block_path="./single_small/single",
                                           output_degree=False,
                                           MPI_rank=i)
        '''
        size = block_threshold[-1]

        single_noise = np.random.rand(500, size).astype(np.float32)
        base = 0
        for i in range(4):
            data = np.load(os.path.join(double_path, "single/block_{}.npz".format(i)))
            property = data["property"]
            np.save(os.path.join(double_path, "single/sample_{}.npy".format(i)),
                    np.ascontiguousarray(single_noise[:, base:base + property.shape[0]]))
            base += property.shape[0]

        assert base == size
        '''

    def _test_random_initialize_for_dti_with_1_big_block(self):
        prob = torch.ones([90, 90]) / 90

        kwords = [{"V_th": -50,
                   "V_reset": -65,
                   'g_Li': 0.03,
                   'g_ui': (0.001617693924345076,
                            0.0001014432345982641,
                            0.003837576135993004,
                            0.000286647496977821),
                   "E_number": int(0.8 * max(5e5 / 90, 0)),
                   "I_number": int(0.2 * max(5e5 / 90, 0))}] * 90

        connect_for_multi_sparse_block(prob, kwords,
                                       degree=1000,
                                       init_min=0,
                                       init_max=1,
                                       perfix="./dti_distribution_500k_randoem/",
                                       dtype=["single"])

        def run(double_path, number):
            os.makedirs(double_path, exist_ok=True)
            block_threshold = merge_dti_distributation_block("./dti_distribution_500k_random/single",
                                                             double_path,
                                                             dtype=["single"],
                                                             number=number,
                                                             debug_block_path="./single_small/single",
                                                             output_degree=True)

            size = block_threshold[-1]

            single_noise = np.random.rand(500, size).astype(np.float32)
            base = 0
            for i in range(number):
                data = np.load(os.path.join(double_path, "single/block_{}.npz".format(i)))
                property = data["property"]
                np.save(os.path.join(double_path, "single/sample_{}.npy".format(i)),
                        np.ascontiguousarray(single_noise[:, base:base + property.shape[0]]))
                base += property.shape[0]

            assert base == size

        run("./dti_single_500k/", 1)
        run("./dti_3_500k/", 3)

    def _test_random_initialize_for_dti_single_block_1_6m_for_verify(self):
        data = np.load("dti_single_1_6m/single/block_0.npz")
        single_idx = data["idx"]
        single_weight = data["weight"]
        data = np.load("dti_double_1_6m/single/block_0.npz")
        double_idx_A = data["idx"]
        double_weight_A = data["weight"]
        data = np.load("dti_double_1_6m/single/block_1.npz")
        double_idx_B = data["idx"]
        double_weight_B = data["weight"]
        resort_idx = np.argsort(np.load("dti_double_1_6m/resort_idx.npy"))

        size = data["property"].shape[0]
        del data

        for ii in np.random.choice(single_idx.shape[1], 10000):
            idx_0, idx_1, idx_2, idx_3 = single_idx[:, ii]
            v = single_weight[ii]
            assert idx_1 == 0
            block_idx = resort_idx[idx_0] // size
            new_idx_0 = resort_idx[idx_0] % size
            new_idx_1 = resort_idx[idx_2] // size
            new_idx_2 = resort_idx[idx_2] % size
            if block_idx == 0:
                select_idx = double_idx_A
                select_weight = double_weight_A
            else:
                select_idx = double_idx_B
                select_weight = double_weight_B
            idx = ((select_idx[0] == new_idx_0)
                   & (select_idx[1] == new_idx_1)
                   & (select_idx[2] == new_idx_2)
                   & (select_idx[3] == idx_3)).nonzero()[0]
            assert idx[0].shape[0] == 1
            idx = idx[0][0]
            print(ii, idx)
            assert select_weight[idx] == v


if __name__ == "__main__":
    unittest.main()
