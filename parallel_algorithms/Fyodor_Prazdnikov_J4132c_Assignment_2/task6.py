from mpi4py import MPI
import time
import sys

# Get my rank
rank = MPI.COMM_WORLD.Get_rank()

NUMBER_OF_ELEMENTS = 50000
SHEET_SIZE = 1000
N = 10


for size in range(1, NUMBER_OF_ELEMENTS + 1, SHEET_SIZE):
    data = [0] * size
    L = sys.getsizeof(data)

    match rank:
        case 0:
            start_time = time.time()
            for _ in range(N):
                MPI.COMM_WORLD.send(data, dest=1, tag=0)
                MPI.COMM_WORLD.recv(source=1, tag=1)
            T = time.time() - start_time

            R = (2 * N * L) / T
            print(
                "Message size: {} bytes with bandwidth {:.2f} MB/s".format(
                    L, R / (1024**2)
                )
            )
        case 1:
            for _ in range(N):
                received_data = MPI.COMM_WORLD.recv(source=0, tag=0)
                MPI.COMM_WORLD.send(received_data, dest=0, tag=1)
