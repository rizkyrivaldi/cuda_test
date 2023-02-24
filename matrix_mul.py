import numpy as np
from numba import cuda, jit
import time

# Controls threads per block and shared memory usage.
# The computation will be done on blocks of TPBxTPB elements.
BPG = 128 # Blocks per grid
TPB = 16 # Threads per block

@jit(nopython=True)
def norm_mul(A, B):
    return A*B

@cuda.jit
def matmul(A, B, C):
    """Perform square matrix multiplication of C = A * B
    """
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp

@cuda.jit
def fast_matmul(A, B, C):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=np.float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=np.float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(bpg):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp


# A = np.random.randint(0, 4000, size = (200, 200))
# B = np.random.randint(0, 4000, size = (200, 200))
A = np.random.rand(4000, 4000)
B = np.random.rand(4000, 4000)
C = np.zeros((200, 200))

A_d = cuda.to_device(A)
B_d = cuda.to_device(B)
C_d = cuda.to_device(C)

# Multiplication
start_time = time.time()
matmul[BPG, TPB](A_d, B_d, C_d)
time_diff_mul = time.time() - start_time

# Fast Multiplication
start_time = time.time()
fast_matmul[BPG, TPB](A_d, B_d, C_d)
time_diff_fmul = time.time() - start_time

# Numpy Multiplication
start_time = time.time()
result = A*B
time_diff_nmul = time.time() - start_time

# Numpy Njit
start_time = time.time()
norm_mul(A, B)
time_diff_njmul = time.time() - start_time

print(time_diff_mul)
print(time_diff_fmul)
print(time_diff_nmul)
print(time_diff_njmul)

print("")

# Multiplication
start_time = time.time()
matmul[BPG, TPB](A_d, B_d, C_d)
time_diff_mul = time.time() - start_time

# Fast Multiplication
start_time = time.time()
fast_matmul[BPG, TPB](A_d, B_d, C_d)
time_diff_fmul = time.time() - start_time

# Numpy Multiplication
start_time = time.time()
result = A*B
time_diff_nmul = time.time() - start_time

# Numpy Njit
start_time = time.time()
norm_mul(A, B)
time_diff_njmul = time.time() - start_time

print(time_diff_mul)
print(time_diff_fmul)
print(time_diff_nmul)
print(time_diff_njmul)

print("")

# Multiplication
start_time = time.time()
matmul[BPG, TPB](A_d, B_d, C_d)
time_diff_mul = time.time() - start_time

# Fast Multiplication
start_time = time.time()
fast_matmul[BPG, TPB](A_d, B_d, C_d)
time_diff_fmul = time.time() - start_time

# Numpy Multiplication
start_time = time.time()
result = A*B
time_diff_nmul = time.time() - start_time

# Numpy Njit
start_time = time.time()
norm_mul(A, B)
time_diff_njmul = time.time() - start_time

print(time_diff_mul)
print(time_diff_fmul)
print(time_diff_nmul)
print(time_diff_njmul)