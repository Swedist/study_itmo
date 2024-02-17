from mpi4py import MPI
import numpy as np

# Get my rank
rank = MPI.COMM_WORLD.Get_rank()


class Database:
    def __init__(self) -> None:
        self._data = {}

    def __str__(self) -> str:
        return str(self._data)

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]


object1 = [1, 2, 3, "hello", "world"]
object2 = Database()

object2[1] = 2
object2["foo"] = "bar"

object3 = np.array(["lorem", "ipsum", "something", 3, 2, 1])

list_of_objects = [object1, object2, object3]
if rank == 0:
    MPI.COMM_WORLD.send(object1, dest=1, tag=1)
    MPI.COMM_WORLD.send(object2, dest=2, tag=2)
    MPI.COMM_WORLD.send(object3, dest=3, tag=3)

match rank:
    case 1:
        message = MPI.COMM_WORLD.recv(source=0, tag=1)
        print(message)
    case 2:
        message = MPI.COMM_WORLD.recv(source=0, tag=2)
        print(message)
    case 3:
        message = MPI.COMM_WORLD.recv(source=0, tag=3)
        print(message)
