"""
这个类中主要保存所用到的网络架构的基本信息，以及进行路由生成时用到的体素连接概率表、体素的大小
1. 网络架构：
    · 昆仑超算中心使用IB（infinite band）架构，IB交换机之间两两相连,交换机之间减速比有0.8
    · 每个IB交换机下有20个计算节点，每个计算节点有4张加速卡（dcu）
2. 输入数据（input）
2.1. 连接概率表（self.conn)
    · 连接概率表是一个约22703x22703的矩阵，第i行的第j个元素代表i号体素连接到j号体素的概率
    · 连接概率表每行元素的和为1
    · 连接概率表是一个稀疏矩阵，只有3%的非0元素（体素版本）
    · 若i号体素连接到j号体素
2.2. 体素的大小（self.size)
    · 记录了每个体素的大小，体素的大小与可能发送的spike数目呈正比
2.3. 体素对外连接的度（self.degree）
    · 记录每个体素与其他体素相连的度

last modified: lyh 2021.1.6
"""
# TODO(@lyh): update documentation


class Base:
    def __init__(self):
        self.neuron_number = int(8.6e10)
        self.N = int(10000)  # Total number of processes
        self.n_gpu_per_group = 100

        self.if_show_progress = False

        # version information
        # self.conn_version = 'voxel_22703'
        self.conn_version = 'cortical_v2'
        self.map_version = 'map_' + str(self.N) + '_split_14_3_' + self.conn_version
        self.route_version = 'route_v1_' + self.map_version

        # old versions
        # self.map_version = 'map_2000_v2'
        # self.route_version = 'route_2000_v1_map_v2'

        # group parameters
        self.n_node_per_group = int(self.n_gpu_per_group / 4)
        self.number_of_groups = int(self.N / self.n_gpu_per_group)

        # data path
        self.root_path = '../'
        self.conn_root = self.root_path + 'tables/conn_table/' + self.conn_version + '/'
        self.conn_table_path = self.conn_root + 'conn.npy'

        self.size_path = self.conn_root + 'size.npy'
        self.origin_size_path = self.conn_root + 'origin_size.npy'

        self.degree_path = self.conn_root + 'degree.npy'
        self.origin_degree_path = self.conn_root + 'origin_degree.npy'

        # network topology information
        self.gpu_per_node = 4
        self.node_per_switch = 20
        self.gpu_per_switch = self.gpu_per_node * self.node_per_switch
        self.number_of_nodes = int(self.N / self.gpu_per_node)
        self.number_of_switches = int(self.number_of_nodes / self.node_per_switch)

        # route path
        self.route_path = self.root_path + 'tables/route_table/' + self.route_version + '/'
        # import os
        # if not os.path.exists(self.route_path):
        #     os.mkdir(self.route_path)
        self.route_npy_save_path = self.route_path + 'route.npy'
        self.route_dense_json_save_path = self.route_path + 'route_dense.json'


if __name__ == "__main__":
    b = Base()
