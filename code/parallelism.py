"""
The class "GenerateMapParallel" generates map table in a parallel way.
"""

from mpi4py import MPI
from base import Base


class Parallelism(Base):
    def __init__(self):
        super().__init__()

        # Initialize MPI
        self.comm = MPI.COMM_WORLD
        self.comm_size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.node_name = MPI.Get_processor_name()

        self.master_rank = self.comm_size - 1

        self.show_initialize_info()
        self.comm.barrier()

    def show_initialize_info(self):
        if self.rank == self.master_rank:
            print('#########################################')
            print('MPI initialize Complete. comm_size = %d' % self.comm_size)
            print('Rank %d is the master rank.' % self.master_rank)
            print('#########################################\n')

    def test_mpi(self):
        print('Hello world from process %d at %s.' % (self.rank, self.node_name))

    # master rank is self.comm_size - 1
    def allocate_idx_to_calculate(self):
        """
        Allocate idxes for processes.
        For example, suppose there are 100 dcus, and 11 processes in total, for the 11th process is responsible for
        gathering results, 10 processes participate in the computing process, which means 1 process is responsible
        for 10 columns' computing. Thus, the columns charged by process k is [k, 10+k, 20+k, ... 90+k].
        """
        assert self.N % (self.comm_size-1) == 0

        idxes = list()
        for idx in range(int(self.N / (self.comm_size-1))):
            idxes.append(idx * (self.comm_size-1) + self.rank)

        return idxes

    def allocate_population_idx_to_calculate(self, n):
        population_indexes = list()
        for idx in range(n):
            if idx % (self.comm_size - 1) == self.rank:
                population_indexes.append(idx)

        return population_indexes


if __name__ == '__main__':
    Job = Parallelism()
