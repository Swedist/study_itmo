from mpi4py import MPI
import numpy as np

# Get my rank and size
size, rank = MPI.COMM_WORLD.Get_size(), MPI.COMM_WORLD.Get_rank()
N = 100000


match rank:
    # Host
    case 0:
        vector_1 = np.ones(N, dtype=int)
        vector_2 = np.full(N, 2, dtype=int)

        message_1 = np.array_split(vector_1, size - 1)
        message_2 = np.array_split(vector_2, size - 1)

        for worker in range(size - 1):
            MPI.COMM_WORLD.send(message_1[worker], dest=worker + 1, tag=worker + 1)
            MPI.COMM_WORLD.send(message_2[worker], dest=worker + 1, tag=worker + 1001)

        dot_product = 0

        for worker in range(size - 1):
            dot_product += MPI.COMM_WORLD.recv(source=worker + 1, tag=worker + 2001)

        result_text = "Dot product of two vectors = {:.3f}".format(dot_product)

        print("_" * len(result_text), result_text, sep="\n")
    # Workers
    case _:
        vector_part_1 = MPI.COMM_WORLD.recv(source=0, tag=rank)
        vector_part_2 = MPI.COMM_WORLD.recv(source=0, tag=rank + 1000)
        result = np.dot(vector_part_1, vector_part_2)
        print("Worker {} sum = {:.3f}".format(rank, result))

        MPI.COMM_WORLD.send(result, dest=0, tag=rank + 2000)
