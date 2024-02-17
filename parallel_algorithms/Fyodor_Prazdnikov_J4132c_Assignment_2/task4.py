from mpi4py import MPI
import time

# Get my rank
rank = MPI.COMM_WORLD.Get_rank()


match rank:
    case 0:
        time.sleep(5)
        data = MPI.COMM_WORLD.recv(source=1, tag=0)
        elapsed_time = 1000 * (time.time() - data["start_time"])
        print(
            'The message "{}" received from rank {} took {:.3f} ms to process.'.format(
                data["message"],
                data["rank"],
                elapsed_time,
            )
        )
    case 1:
        data = {
            "start_time": time.time(),
            "message": "Hello, World!",
            "rank": rank,
        }
        MPI.COMM_WORLD.send(data, dest=0, tag=0)
        n = 10
        for i in range(n):
            print("Work in progress [{}/{}]".format(i + 1, n))
            time.sleep(0.25)
