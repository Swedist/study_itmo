from mpi4py import MPI
import time

# Get my rank and size
size, rank = MPI.COMM_WORLD.Get_size(), MPI.COMM_WORLD.Get_rank()

match rank:
    case 0:
        for i in range(1, size):
            data = MPI.COMM_WORLD.recv(source=i, tag=0)
            elapsed_time = 1000 * (time.time() - data["start_time"])
            print(
                'The message "{}" received from rank {} took {:.3f} ms to process.'.format(
                    data["message"],
                    i,
                    elapsed_time,
                )
            )
    case _:
        data = {
            "start_time": time.time(),
            "message": "Hello, World!",
        }
        MPI.COMM_WORLD.send(data, dest=0, tag=0)
