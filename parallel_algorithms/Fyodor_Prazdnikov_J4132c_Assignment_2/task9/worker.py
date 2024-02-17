from mpi4py import MPI
import random

comm = MPI.Comm.Get_parent()
rank = comm.Get_rank()

print("Worker's rank: {}".format(rank))

data = {
    "rank": rank,
    "message": "Hello, World! Random number is {}".format(random.randint(1, 100)),
}
comm.send(data, dest=0, tag=0)

comm.Disconnect()
