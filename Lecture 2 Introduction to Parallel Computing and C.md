# Lecture 2: Introduction to Parallel Computing and CUDA

Reviewed: Yes
Textbook Chapter: 2

# Background

- **data parallelism**: idea that computational work to be performed on different parts of the datasets can be done independently of/in parallel with each other

# 2.1 Data parallelism

- writing data parallel code entails (re)organizing the computation around the data such that they can be done independently and in parallel, completing the overall job must faster
- eg. converting an image (denoted as an array $I$ of RGB values) to grayscale (denoted as an array $O$ of RGB values) requires just doing calculations on each pixel independently because they don’t depend on each other
    
    ![Screenshot 2024-09-06 at 6.03.41 PM.png](files/Lecture%202%20Introduction%20to%20Parallel%20Computing%20and%20C/Screenshot_2024-09-06_at_6.03.41_PM.png)
    

## Task Parallelism vs. Data Parallelism

- both are types of parallelisms used in parallel programming
- **task parallelism**: entails decomposing the *tasks* of applications → eg. if an application needs to do both a vector addition task and a matrix-vector multiplication task, task parallelism exists if the two tasks can be done independently
    - common sources of tasks that can undergo task parallelism are I/O and data transfers
- data parallelism is generally the main source of scalability for parallel programs

# 2.2 CUDA C program structure

- CUDA C extends the traditional ANSI C programming language and is built on NVIDIA’s CUDA platform
- structurally, there is a **host (CPU)** and one or more **devices (GPU)** in the computer, and each CUDA C source file can have a mixture of host code and device code
    - device code is marked with special CUDA C keywords and includes **kernels** → functions where the code is executed in a data-parallel manner

## CUDA program execution steps:

1. Starts with host code (CPU serial code) calling a kernel function.
2. When a kernel function is called, a large number of threads (collectively called a **grid**) are launched on a device to execute the kernel.
    1. When launching a grid of threads, each thread needs to be generated → can assume that these threads take very few clock cycles to generate and schedule due to efficient hardware.
        1. This efficiency is contrasted with traditional CPU threads which typically take thousands of clock cycles to generate and schedule
3. When all the threads of a grid have completed, the grid terminates and the execution continues on the host until another grid is launched.

![Note: this is a simplified model in which the CPU execution and GPU execution don’t overlap, but many heterogeneous computing applications manage overlapped CPU and GPU execution to take advantage of both CPUs and GPUs.](files/Lecture%202%20Introduction%20to%20Parallel%20Computing%20and%20C/Screenshot_2024-09-07_at_10.06.15_PM.png)

Note: this is a simplified model in which the CPU execution and GPU execution don’t overlap, but many heterogeneous computing applications manage overlapped CPU and GPU execution to take advantage of both CPUs and GPUs.

# 2.3 A vector addition kernel

- vector addition is the simplest possible data parallel computation
- traditional sequential C implementation

```c
// Compute vector sum C_h = A_h + B_h
void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
		for (int i = 0; i < n; ++i) {
				C_h[i] = A_h[i] + B_h[i];
		}
}

int main() {
		// Memory allocation for arrays A, B, and C
		// I/O to read A and B, N elements each
		...
		vecAdd(A, B, C, N); // passing the pointers to these arrays to vecAdd
}

// _h denotes variables used by the host
// _d denotes variabels used by the device
```

## C Pointers Recap

- pointers ****used to access variables and data structures, declared with `float *P`
- `P = &V` assigns the address of `V` to `P` → `P` points to `V`
- `*P` gets the value of the object at the address pointer `P` stores, namely `V`
- An array in a C program can be accessed through a pointer that points to its 0th element
    - `P = &(A[0])` makes `P` point to the 0th element of array `A` → `P[i]` becomes a synonym for `A[i]`
    - the array name `A` is in itself a pointer to its 0th element
    - in the example above, passing an array name `A` as the first argument to function call to `vecAdd` makes the function’s first parameter `A_h` point to the 0th element of `A`. As a result, `A_h[i]` in the function body can be used to access `A[i]` for the array `A` in the main function

## Parallelizing the Vector Addition Code

- this host code is a **stub** for calling a kernel

```c
void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
		int size = n * sizeof(float);
		float *A_d, *B_d, *C_d;
		
		// Part 1: Allocate device memory for A_d, B_d, and C_d
		// Copy A_h and B_h TO device memory
		...
		
		// Part 2: Call kernel to launch a grid of threads
		// to perform actual vector addition
		...
		
		// Part 3: Copy C_d FROM device memory
		// Deallocate A_d, B_d, and C_d vectors from device memory
		...
}
```

![Note: In practice, this model can be very inefficient because of all the copying of data back and forth. Usually you’d keep large and important data structures on the device and simply have the host code invoke device functions on them.](files/Lecture%202%20Introduction%20to%20Parallel%20Computing%20and%20C/Screenshot_2024-09-07_at_10.18.15_PM.png)

Note: In practice, this model can be very inefficient because of all the copying of data back and forth. Usually you’d keep large and important data structures on the device and simply have the host code invoke device functions on them.

# 2.4 Managing device global memory and data transfer

- in current CUDA systems, the devices are often hardware cards that have their own DRAM called device global memory → calling it “global” memory distinguishes it from the other types of device memory (more details in Chapter 5)
- recall Part 1 and Part 3 in the previous example where host program needs to (part 1) allocate space in device global memory and transfer data from host to said allocated space, and (part 3) transfer the result data from device global memory back to host memory and deallocate the space

## Managing Device Global Memory

- CUDA runtime system (typically running on the host) provides API functions to do this:
    
    ```c
    // Example Usage
    
    float *A_d;
    int size = n * sizeof(float);
    cudaMalloc( (void **)&A_d, size);
    ...
    cudaFree(A_d);
    ```
    
    1. `cudaMalloc()` : allocates object in the device global memory
        - 2 parameters:
            - **address of the pointer variable to the allocated object** → this should be able to be cast to `void **` because the functions expects a generic pointer
            - **size** of the allocated object in terms of bytes
        - this function writes the address of the allocated memory (in the device global memory) in the provided pointer variable → the host code that calls the kernels then passes this provided pointer value (now written with the value of the address of allocated memory) to the kernels that need access to the allocated memory object
            - TLDR: on return, the provided pointer parameter will point to the device globabl memory region allocated
        - Note: this `cudaMalloc` has a different format from the C `malloc` function
            - C `malloc` function returns a pointer to the allocated object and takes only one parameter that specifies the size of the allocated object
            - `cudaMalloc` function writes to the pointer variable whose address is given as the first parameter, and the two parameter format allows it to use the return value to report any errors in the same way as other CUDA API functions
    2. `cudaFree()` : frees object from the device global memory
        - 1 parameter:
            - **pointer to the freed object**
        - Note: `cudaFree` does not need to change the value of `A_d`; it only needs to use the value of `A_d` to return the allocated memory back to the available pool → thus only the value and not the address `*A_d` is passed as an argument
- the addresses that are cudaMalloced (eg. `A_d`, `B_d`, `C_d`) point to locations in the device global memory and **should not be dereferenced in the host code** → should be used in calling API functions and kernel functions
    - dereferencing a device global memory pointer in host code can cause exceptions or other types of runtime errors

## Data Transfer

- accomplished by calling one of the CUDA API functions
    
    ```c
    // Example Usage
    
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);
    ...
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
    
    // cudaMemcpyHostToDevice and cudaMemcpyDeviceToHost are symbolic constants
    // that are recognized and predefined constants in the CUDA programming env
    ```
    
    1. `cudaMemcpy()`: memory data transfer
        - 4 parameters:
            - pointer to destination location
            - point to source location
            - number of bytes to be copied
            - types of memory involved in the copy (noted by the constants):
                - from host to host
                - from host to device
                - from device to host
                - from device device to device

## Expanding on the Vector Addition Parallelization

```c
void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
		int size = n * sizeof(float);
		float *A_d, *B_d, *C_d;
		
		// Part 1: Allocate device memory for A_d, B_d, and C_d
		cudaMalloc((void **) &A_d, size); 
		cudaMalloc((void **) &B_d, size); 
		cudaMalloc((void **) &C_d, size); 
		// Copy A_h and B_h TO device memory
		cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice); 
		cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice); 
		
		// Part 2: Call kernel to launch a grid of threads
		// to perform actual vector addition
		...
		
		// Part 3: Copy C_d FROM device memory
		cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost); 
		// Deallocate A_d, B_d, and C_d vectors from device memory
		cudaFree(A_d); 
		cudaFree(B_d); 
		cudaFree(C_d); 
}
```

## Error Checking and Handling in CUDA

- CUDA API functions return flags that indicate whether an error has occurred when a request has been served

```c
// Example CUDA Error Checking and Handling

cudaError_t err = cudaMalloc((void **) &A_d, size);
if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
}
```

# 2.5 Kernel functions and threading

- **kernel function**: specifies the code to be executed by ALL threads during a parallel phase
    - all these threads execute the SAME code, which makes CUDA C programming an instance of the single-program multiple-data (SPMD) parallel programming style

![Example where there are N blocks each containing 256 threads, organized as a 1D array of threads for 1D vector data](files/Lecture%202%20Introduction%20to%20Parallel%20Computing%20and%20C/Screenshot_2024-09-08_at_1.17.09_AM.png)

Example where there are N blocks each containing 256 threads, organized as a 1D array of threads for 1D vector data

- when host code calls a kernel, the CUDA runtime systems launches a grid of threads organized in a two-level hierarchy
    - each grid is organized as an array of **thread blocks** (aka **blocks**) → all blocks of a grid are the same size, which can go up to 1024 threads (on current systems 3.0 and beyond at least)
        - the ****size of kernel is specified by the host code and the same kernel can be called with different thread block sizes/number of threads at different parts of the host code
        - **`blockDim`**: built-in variable, struct with 3 unsigned integer fields (x, y, z) that organizes the threads into 1D, 2D, or 3D arrays (eg. for 1D organization, only the x field is used, etc.) → choice of dimensionality for organizing threads usually reflects the dimensionality of the data
            - use `blockDim.x`, `blockDim.y`, `blockDim.z` variables to access the number of threads in the particular dimension
            - generally recommended for the number of threads in each dimension to be a multiple of 32 for hardware efficiency reasons
- CUDA kernels also have access to 2 more built-in variables that allow threads to distinguish themselves from each other and to determine the area of data each thread is to work on → this is the two-level hierarchy mentioned before that offers a form of locality
    - **`threadIdx`**: gives each thread a unique coordinate in a dimension within a block
        - can use `threadIdx.x`, `threadIdx.y`, `threadIdx.z`
        - eg. in the example above, 0 to 255 would be the range of `threadIdx.x` (and only using the x dimension because it is a 1D array of threads)
    - **`blockIdx`**: gives all threads in a block a common block coordinate
        - can use `blockIdx.x`, `blockIdx.y`, `blockIdx.z`
        - eg. in the example above, 0 to N-1 would be the range of `blockIdx.x`
    - a unique global thread index (for the 1D example) can be calculated as:
        - `i = blockIdx.x * blockDim.x + threadIdx.x`
        - eg. if `blockDim = 256`, the `i` values of threads in block 0 range from 0 to 255, values for threads in block 1 range from 266 to 511, and the values for threads in block 2 range from 512 to 767 → so these 3 blocks form a continuous coverage of 768 iterations of the original loop accessing the vectors `A`, `B`, and `C`
            - by launching a grid with `n` or more threads, one can process vectors of length `n`
    - these are means for the threads to access hardware registers that provide the identifying coordinates to threads
- `i` is an **automatic local variable** that is private to each thread and not visible to other threads

## Writing a kernel function

```c
// Compute vector sum C = A + B
// Each thread performs one pair-wise addition

__global__
void vecAddKernel(float* A, float* B, float* C, int n) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		// This conditional is here because not all vector lengths can be
		// expressed as multiples of the block size.
		// Eg. Consider a vector length n = 100. 
		// The smallest efficient thread block dimension is 32, and
		// assume we picked 32 as the block size.
		// One would need to launch 4 thread blocks (i.e. 128 threads)
		// to process all n = 100 elements.
		// Since there is an extra 28 threads that i will have the value of,
		// we need to disable them from doing work not expected by the original
		// program (i.e. A[101] is not accessible)
		// This catch allows the kernel to be called to process vectors of
		// arbitrary lengths.
		if (i < n) {
				C[i] = A[i] + B[i];
		}
}
```

- this is executed all on the device
- `__global__` keyword indicates that the function is a kernel and that it can be called to generate a new grid of threads on a device
    - executed on the device and can be called from the host → in CUDA systems that support *dynamic [parallelism](https://www.sciencedirect.com/topics/computer-science/parallelism)*, it can also be called from the device
- other keywords:
    - `__device__`: indicates that the function being declared is a **CUDA device function**: function that executes on a CUDA device and can only be called from a kernel function or another device function
        - TLDR: access restricted to device
        - does not result in any new device threads being launched
    - `__host__`: indicates that the function being declared is a **CUDA host function**: just a traditional C function that executes on the host and can only be called from another host function
    - can use both `__device__` and `__host__` in a single function declaration → this combination tells the compilation system to generate 2 versions (one to be executed on the host and only callable from a host function, one to be executed on the device and only callable from a device or kernel function) of the object code from the same function
    
    ![Screenshot 2024-09-08 at 2.34.18 AM.png](files/Lecture%202%20Introduction%20to%20Parallel%20Computing%20and%20C/Screenshot_2024-09-08_at_2.34.18_AM.png)
    
- Note: there is no loop going over the length n of the vectors → this loop has been replaced with the grids of threads, where each thread corresponds to one iteration of the original loop
    - this is sometimes referred to as **loop parallelism**: iterations of the original sequential code are executed by threads in parallel

# 2.6 Calling kernel functions

- last step after writing the kernel function is to just call it from the host code to launch the grid

```c
// Example of kernel calling
int vectAdd(float* A, float* B, float* C, int n) {
		// A_d, B_d, C_d allocations and copies omitted
		...
		// Launch ceil(n/256) blocks of 256 threads each
		vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);
} 
```

- host code sets the grid and block dimensions via **execution configuration parameters** given as `<<< number of blocks in the grid, number of threads in each block>>>` before the function arguments
    - in the example, number of blocks in grid = ceil(number of threads/ thread block size) to ensure that there are enough threads in the grid to cover all the vector elements (and using floating point to ensure that the division generates a floating value so that the ceil function can round it up correctly)

## Completing the Host Code for Vector Addition Parallelization

```c
void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
		int size = n * sizeof(float);
		float *A_d, *B_d, *C_d;
		
		// Part 1: Allocate device memory for A_d, B_d, and C_d
		cudaMalloc((void **) &A_d, size); 
		cudaMalloc((void **) &B_d, size); 
		cudaMalloc((void **) &C_d, size); 
		// Copy A_h and B_h TO device memory
		cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice); 
		cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice); 
		
		// Part 2: Call kernel to launch a grid of threads
		// to perform actual vector addition
		vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);
		
		// Part 3: Copy C_d FROM device memory
		cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost); 
		// Deallocate A_d, B_d, and C_d vectors from device memory
		cudaFree(A_d); 
		cudaFree(B_d); 
		cudaFree(C_d); 
}
```

- Note: all the thread blocks operate on different parts of the vectors, and can be executed in any arbitrary order → so can NOT make any assumptions regarding execution order
- Important Note: this vector addition example is used for simplicity → in practice the overhead of allocating device memory, input data transfer from host to device, output data transfer from device to host, and deallocating device memory will likely make the resulting code slower than the original sequential code
    - this is because the amount of calculation that is done by the kernel is small relative to the amount of data processed or transferred
        - REAL, EFFECTIVE, WORTHWHILE APPLICATIONS OF PARALLELIZATION typically have kernels where there is relatively more computational work than the amount of data processed, and also tend to keep the data in the *device* memory across multiple kernel invocations so that the overhead can be amortized

# 2.7 Compilation

- implementing CUDA C kernels requires using various extensions that render the resulting code incompatible with a traditional C compiler
- NVCC (NVIDIA C compiler): a compiler that recognizes and understands the extensions
    - processes a CUDA C program by using the CUDA keywords to separate the host and device code
        - the host code is pure ANSI C code → compiled with the host’s standard C/C++ compilers and run as a traditional CPU process
        - device code → compiled by NVCC into virtual binary files called PTX files → PTX files are then further compiled by a runtime component of NVCC into the real object files and executed on a CUDA-enabled GPU device

![Screenshot 2024-09-10 at 2.34.58 PM.png](files/Lecture%202%20Introduction%20to%20Parallel%20Computing%20and%20C/Screenshot_2024-09-10_at_2.34.58_PM.png)

# 2.8 Summary

## Function declarations

- using one of `__global__`, `__device__`, `__host__`, a CUDA C programmer can instruct the compiler to generate a kernel function, a device function, or a host function.
    - All function declarations without any of these keywords default to host functions.
    - If both `__host__` and `__device__` are used in a function declaration, the compiler generates two versions of the function: one for the device and one for the host
    - If a function declaration does not have any CUDA C extension keyword, the function defaults into a host function

## Kernel call and grid launch

- CUDA C extends the C function call syntax with kernel execution configuration parameters `<<< number of blocks in the grid, number of threads in each block>>>`
    - there are other types of execution configuration parameters

## Built-in (predefined) variables

- CUDA kernels can access a set of built-in, predefined, read-only variables that allow each thread to distinguish itself from other threads and to determine the area of data to work on
    - `threadIdx`
    - `blockDim`
    - `blockIdx`

## Runtime API

- CUDA supports a set of API functions to provide services to CUDA C programs.
    - `cudaMalloc`
    - `cudaFree`
    - `cudaMemcpy`

# Resources

- Kirk & Hwu: Chapter 2
- NVIDIA CUDA Docs: [https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model)
- Lecture Slides:
- Lecture Video: