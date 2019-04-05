from mpi4py import MPI


def print_rank0(comm, str):
    if comm.Get_rank() == 0:
        lines = str.split('\n')
        prefix = '{:02d}: '.format(MPI.COMM_WORLD.Get_rank())
        str = ('\n'+prefix).join(lines)
        str = prefix+str
        print(str)


def print_all_ranks(comm, str):
    print("Rank {}({}): ".format(comm.Get_rank(), MPI.COMM_WORLD.Get_rank())+str)
