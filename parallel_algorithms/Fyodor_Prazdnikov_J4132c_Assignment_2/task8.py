from mpi4py import MPI
import time

# Get my rank
rank = MPI.COMM_WORLD.Get_rank()

HOST_SLEEP_TIME = 25


match rank:
    # Host
    case 0:
        message = "Hello from host"
        req = MPI.COMM_WORLD.isend(message, dest=1, tag=0)

        for _ in range(HOST_SLEEP_TIME // 5):
            time.sleep(5)
            print("WAITING")

        req.wait()
        print("Message sent")
    # Receiver
    case 1:
        message = MPI.COMM_WORLD.recv(source=0, tag=0)
        print('Receiver {} got message: "{}"'.format(rank, message))
