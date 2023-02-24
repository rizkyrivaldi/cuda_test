import numpy as np
from numba import cuda
import time

BPG = 128 # Blocks per grid
TPB = 16 # Threads per block

_TPB = (TPB, TPB)

i = 0

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        print(f"Time elapsed: {time.time() - start} seconds")
    return wrapper

# kwargs (fastmath=True/False, parallel=True/False)

@cuda.jit
def increment(A):
    row, col = cuda.grid(2)
    if row < A.shape[0] and col < A.shape[1]:
        A[row, col] = A[row, col]**2
    cuda.syncthreads()

def increment_raw(A):
    row, col = A.shape
    for i in range(row):
        for j in range(col):
            A[i, j] ** 2
    return A

@cuda.jit
def checkGrid():
    row, col = cuda.grid(2)
    print(col)

# checkGrid[BPG, TPB]()

A_Matrix = np.array([[1, 2, 3], [4, 5, 6]])

B_Matrix = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
])

C_Matrix = np.random.randint(0, 10, size=(2000, 2000))

C_Matrix = np.linspace((1,2),(10,20),10)
# A_Matrix = [[[1, 2, 3],[4, 5, 6]]]
# print(A_Matrix.shape[1])

operand = C_Matrix
# print(operand)
# print("After increment")
start_time = time.time()
passed = cuda.to_device(operand)
increment[BPG, _TPB](passed)
operand = passed.copy_to_host()
time_diff_gpu = time.time() - start_time
# print(operand)

start_time = time.time()
increment_raw(operand)
time_diff = time.time() - start_time

print(f"Time elapsed without GPU : {time_diff} second")
print(f"Time elapsed with GPU    : {time_diff_gpu} second")


