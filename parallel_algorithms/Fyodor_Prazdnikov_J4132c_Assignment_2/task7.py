from mpi4py import MPI

# Get my rank and size
rank, size = MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size()


match rank:
    # Host
    case 0:
        message = "Hello from host"
        MPI.COMM_WORLD.send(message, dest=1)

        original_message = MPI.COMM_WORLD.recv(source=size - 1)
        print('Host received original message "{}"\nDONE'.format(original_message))
    # Workers
    case _:
        message = MPI.COMM_WORLD.recv(source=(rank - 1) % size)
        if rank == 1:
            from_ = "Host"
        else:
            from_ = "Worker {}".format(rank - 1)
        print('Worker {}: message from {} "{}"'.format(rank, from_, message))
        MPI.COMM_WORLD.send(message, dest=(rank + 1) % size)
