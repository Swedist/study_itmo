from mpi4py import MPI
import sys

N = 10
comm = MPI.COMM_WORLD.Spawn(sys.executable, args=["worker.py"], maxprocs=N)


for _ in range(N):
    data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
    print('Got message "{}" from Worker {}'.format(data["message"], data["rank"]))

comm.Disconnect()
