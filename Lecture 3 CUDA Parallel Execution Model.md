# Lecture 3: CUDA Parallel Execution Model

# Background

- previous chapter showed an example with data elements of (and thus threads for) a 1D array
- this content looks more generally at how threads are organized and how they can be used with MULTIdimensional arrays
- then will explore concept of flexible resource assignment and occupancy, then thread scheduling, latency tolerance, and synchronization

# CUDA Multidimensional grid organization

- RECAP:
    - all threads in a grid execute the same kernel function
    - all threads rely on coordinates (i.e. thread indices) to distinguish themselves from each other and to identify the appropriate portion of the data to process
    - threads are organized into a two-level
    hierarchy:
        - a grid consists of one or more blocks
        - each block consists of one or more threads
            - all threads in a block share the same block index, accessed via the built-in `blockIdx` variable
            - each thread also has a thread index, accessed via the built-in `threadIdx` variable
    - execution configuration parameters in a kernel call statement specify the dimensions of the grid with the built-in `gridDim` variable, and the dimensions of each block, with the built-in `blockDim` variable

## General configurations

- **grid (in general)**: a 3D array of blocks, where each block is a 3D array of threads
    - dimensions are specified in the execution configuration parameters (within `<<<` and `>>>`) of the kernel call with a **type** **dim3** (an integer vector type of 3 elements)
    - all blocks in a grid have the SAME dimensions and sizes
    - total size of ONE block in current CUDA systems is limited to 1024 threads, which can be distributed across the 3 dimensions in any way
        - eg. `blockDim` values of (512, 1, 1) = 512 threads, (8, 16, 4) = 512 threads, (32, 16, 2) = 1024 threads are allowed, but (32, 32, 2) = 2048 threads is not allowed
- **allowed # of blocks**: values of `gridDim.x` $\in [1, 2^{31}-1]$ and allowed values of `gridDim.y` and `gridDim.z` $\in [1, 2^{16}-1]$
    - so among blocks:
        - `blockIdx.x` $\in [0, \text{gridDim.x - 1}]$
        - `blockIdx.y` $\in [0, \text{gridDim.y - 1}]$
        - `blockIdx.z` $\in [0, \text{gridDim.z - 1}]$
    - all threads in a block share the same `blockIdx.x`, `blockIdx.y`, `blockIdx.z` values
- a grid and its blocks don’t need to have the same dimensionality
    
    ![Screenshot 2024-09-10 at 10.08.11 PM.png](files/Lecture%203%20CUDA%20Parallel%20Execution%20Model/Screenshot_2024-09-10_at_10.08.11_PM.png)
    
    - in the above, `gridDim` = (2, 2, 1) (the 4 blocks organized into a 2x2 array), and `blockDim` = (4, 2, 2)
    - in the above, each block is labeled with `(blockIdx.y, blockIdx.x)` so horizontally, each column makes up an x and vertically each row makes up a y → **note the reversed order where it goes z then y then x**
        - the same convention goes for `threadIdx` → eg. thread (1, 0, 2) has `threadIdx.z` = 1, `threadIdx.y` = 0, and `threadIdx.x` = 2

## 1D configuration example

- if fewer than 3 dimensions are used, the size of the unused dimension is by default 1 in the dim3 constructor

```c
// Example host code that generates a 1D grid with:
// 32 blocks
// 128 threads per block
// for a total of 32 * 128 = 4096

dim3 dimGrid(32, 1, 1); // host code variable
dim3 dimBlock(128, 1, 1); // host code variable
vecAddKernel<<<dimGrid, dimBlock>>>(...);
```

- within the kernel function, the `x` field of built-in variables `gridDim` and `blockDim` are preinitialized to the values of the execution configuration parameters

# Mapping threads to multidimensional data

- choice of 1D, 2D, or 3D thread organization is usually based on the nature of the data
    - **Note: will be referring to the dimensions of multidimensional data in descending order, i.e. z dimension, then y dimension, then z dimension**

![Assuming we use a 16x16 block (16 threads in x direction, 16 threads in y direction), we would need ceil(62/16) = 4 blocks for the y direction and ceil(76/16) = 5 blocks in the x direction to cover all pixels.](files/Lecture%203%20CUDA%20Parallel%20Execution%20Model/Screenshot_2024-09-11_at_6.46.38_PM.png)

Assuming we use a 16x16 block (16 threads in x direction, 16 threads in y direction), we would need ceil(62/16) = 4 blocks for the y direction and ceil(76/16) = 5 blocks in the x direction to cover all pixels.

- each thread is assigned to process a pixel with coordinates:
    - y-coord (vertical row) = `blockIdx.y` * `blockDim.y` +  `threadIdx.y`
    - x-coord (horizontal column) = `blockIdx.x` * `blockDim.x` + `threadIdx.x`
- assume that host code defines the following:
    - `n` = number of pixels in the y direction
    - `m` = number of pixels in the x direction
    - `Pin_d` = pointer variable to access the input picture data that has been copied to device global memory
    - `Pout_d` = pointer variable to access the output picture, which has already been allocated on the device memory
- then can use the following host code to call a 2D kernel to process the picture into grayscale
    
    ```c
    dim3 dimGrid(ceil(m/16.0), ceil(n/16.0), 1); // recall constructor is dim3(x,y,z)
    dim3 dimBlock(16, 16, 1);
    colorToGrayscaleConversion<<<dimGrid, dimBlock>>>(Pin_d, Pout_d, m, n);
    ```
    

## How C statements access elements of dynamically allocated multidimensional arrays

- ideally: access `Pin_d` as a 2D array via `Pin_d[j][i]` for the element at row `j` and column `i`
- however: the ANSI C standard that CUDA C is based on requires the number of **columns** in `Pin` to be known at compile time for it to be accessed as a 2D array → **this is not known at compile time for dynamically allocated arrays**
    - **as a result, current CUDA C needs programmers to explicitly linearize a dynamically allocated 2D array into an equivalent 1D array**
- in reality, all multidimensional arrays in C are linearized because modern computers use a “flat” memory
    - for **statically allocated arrays:** compilers allow programmers to use higher-dimensional indexing syntax like `Pin_d[j][i]` to access elements, but under the hood, the compiler linearizes them into an equivalent 1D array and translates the multidimensional indexing syntax into a 1D offset
    - for **dynamically allocated arrays:** CUDA C compiler doesn’t do linearization, programmers need to do the translation because there is the lack of dimensional information at compile time

### Memory Space

- **memory space**: simplified view of how a processor accesses its memory in modern computers
- one memory space is usually associated with each running applications → the data to be processed and the instructions executed are stored in locations in the memory space
    - each location can accommodate a byte and has an address
    - variables requiring multiple bytes (eg. 4 bytes for a float, 8 bytes for a double, etc.) are stored in consecutive byte locations
    - when accessing a data value from the memory space, the processor gives the address of the starting byte and the number of bytes needed

### 2 ways for linearizing a 2D array

1. **row-major layout:** Place all elements of the same row into consecutive locations, and place rows one after another in the memory space
    
    ![Screenshot 2024-09-11 at 11.08.12 PM.png](files/Lecture%203%20CUDA%20Parallel%20Execution%20Model/Screenshot_2024-09-11_at_11.08.12_PM.png)
    
    - denote $M_{j,i}$ as the element of $M$ at the jth row and ith column
    - this is the way C compilers linearize 2D arrays
2. **column-major layout:** Place all elements of the same column in consecutive locations, and place columns one after another in the memory space
    1. equivalent to the row-major layout of the transposed 2D array
    - this is used by FORTRAN compilers

## Kernel code for multidimensional (2D) data

```c
// The input image is encoded as unsignd chars [0, 255]
// Each pixel is 3 consecutive chars for the 3 (RGB) channels

__global__
void colorToGrayscaleConversion(unsigned char* Pout,
																unsigned char* Pin,
																int width,
																int height) {
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		
		if (col < width && row < height) {
				// Get 1D offset for the grayscale image
				int grayOffset = row * width + col;
				
				// One can think of the RGB image having CHANNEL times more columns
				// than the grayscale image
				// assume CHANNELS = 3 is defined outside the kernel
				
				// need to multiply the gray pixel index by 3 because each colored pixel
				// is stored as 3 elements, each of which is 1 byte stored in 3 consecutive
				// locations in the memory space
				int rgbOffset = grayOffset * CHANNELS;
				unsigned char r = Pin[rgbOffset];
				unsigned char g = Pin[rgbOffset + 1];
				unsigned char b = Pin[rgbOffset + 2];
				
				// Perform the rescaling (just an equation) and store it
				Pout[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
		}
}
```

![The result of running the kernel code above, which generates 64 x 80 threads. The grid has 4 x 5 = 20 blocks.](files/Lecture%203%20CUDA%20Parallel%20Execution%20Model/Screenshot_2024-09-11_at_11.25.14_PM.png)

The result of running the kernel code above, which generates 64 x 80 threads. The grid has 4 x 5 = 20 blocks.

- the execution behavior of the 20 blocks will fall into one of 4 cases, depending on how they satisfy the `(col < width && row < height)` condition
    - area 1: both `col` and `row` values are within the range
        - all 16 x 16 = 256 threads in these 16 blocks will process pixels
    - area 2: `row` values are within range, but some `col` values exceed the `m` (width) value
        - the smallest multiple of 16 needed to cover the width of 76 pixels is 80 → so (80 - 76) = 4 unused threads in each row
        - overall, (16 - 4) x 16 = 192 of the 16 x 16 = 256 threads in each of these 3 blocks will process pixels
    - area 3: `col` values are within range, but some `row` values exceed the `n` (height) value
        - the smallest multiple of 16 needed to cover the width of 62 pixels is 64 → so (64 - 62) = 2 unused threads in each column
        - overall, (16 - 2) x 16 = 224 of the 16 x 16 = 256 threads in each of these 4 blocks will process pixels
    - area 4: both of some `col` and `row` values are not within range
        - from above, 4 columns in each of the first 14 rows will be unused and 2 rows in each of the first 12 columns will be unused
        - overall, only 14 x 12 = 168 of the 16 x 16 threads in this block will process pixels

## Extending to 3D data

- include another dimension when linearizing the array → place each “plane” (each 2D array) of the array one after another into the address space
- in the kernel, the array index will involve another global index `int plane = blockIdx.z * blockDim.z + threadIdx.z`
- the linearized access to a 3D array `P` will be in the form `P[**(plane * width * height)** + (row * width) + (col)]` → then kernel also needs to check whether all 3 global indices are in the valid range of the array
- 3D arrays in CUDA kernels are particularly useful for the stencil pattern

# Image blur: a more complex kernel

- **image blurring**: smoothes out abrupt variation of pixel values while preserving the edges that are essential for recognizing the key features of the image
    
    ![Screenshot 2024-09-12 at 12.07.37 AM.png](files/Lecture%203%20CUDA%20Parallel%20Execution%20Model/Screenshot_2024-09-12_at_12.07.37_AM.png)
    
    - mathematically, value of an output image pixel = the weighted sum of a patch of pixels encompassing the said pixel in the input image → this is an example of a **convolution** pattern
- in this example, the image blurring function will be simplified to a simple average of the NxN patch of pixels surrounding (and including) the target pixel (i.e. no weights)

## Kernel code for image blurring

```c
// Each thread calculates an output pixel as before
__global__
void blurKernel(unsigned char* in, unsigned char* out, int width, int height) {
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		
		if (col < width && row < height) {
				int pixVal = 0;
				int pixels = 0;
				
				// Get average of the surrounding BLUR_SIZE x BLUR_SIZE box
				// Assume BLUR_SIZE defined outside of kernel such that 2*BLUR_SIZE gives the number
				// of pixels on each side of the patch (eg. for a 3x3 patch, BLUR_SIZE = 1, for
				// a 7x7 patch, BLUR_SIZE = 3)
				for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++ blurRow) {
						for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++ blurCol) {
								int curRow = row + blurRow;
								int curCol = col + blurCol;
								
								// Verify the image pixel to be added to the average is valid (boundary conditions)
								if (curRow >=0 && curRow < height && curCol >= 0 && curCol < width) {
										pixVal += in[curRow * width + curCol];
										++pixels;
								}
						}
				}
				
				// Write new pixel value to the output image
				out[row * width + col] = (unsigned char) (pixVal / pixels);
		}
}
```

![An example with a 3x3 patch.](files/Lecture%203%20CUDA%20Parallel%20Execution%20Model/Screenshot_2024-09-21_at_3.19.35_PM.png)

An example with a 3x3 patch.

![Screenshot 2024-09-21 at 3.23.48 PM.png](files/Lecture%203%20CUDA%20Parallel%20Execution%20Model/Screenshot_2024-09-21_at_3.23.48_PM.png)

# Synchronization and transparent scalability

- RECAP: have discussed how to launch a kernel for execution by a grid of threads and how to map threads to parts of the data structure
- now discussing how to coordinate the execution of multiple threads
- CUDA allows threads in the same block to coordinate their activities using the **barrier synchronization function `__syncthreads()`**
    - when a thread calls this function, it will be stopped at the program location of the call until EVERY thread in the same block reaches that location as well → ensures that ALL threads in the same block has reached the same location (i.e. completed a phase of their execution task)
        - so the program location of the `__syncthreads()` call is where they will all stop and resume (remember they are all running the same code, so naturally this makes sense)
    - barrier synchronization is a simple and popular method for coordinating parallel activities → TLDR wait for everyone to reach the same point (the barrier) before proceeding
        
        ![Screenshot 2024-09-20 at 4.21.55 PM.png](files/Lecture%203%20CUDA%20Parallel%20Execution%20Model/Screenshot_2024-09-20_at_4.21.55_PM.png)
        
    - in CUDA, if a `__syncthreads()` statement is present, it MUST be executed by ALL threads in a block
        - consider when a `__syncthreads()` statement is placed in an `if-else` statement: we want either ALL threads in the block to enter the same condition path and execute the barrier synchronization, or NONE of the threads to do so
            - so if there exists a `__syncthreads()` call in both the `then` path AND the `else` path, we want ALL threads in the block to execute the `then` path OR the `else` path, which lead to 2 different barrier synchronization points
            - now consider this code:
                
                ```c
                void incorrect_barrier_example(int n) {
                		...
                		if (threadIdx.x % 2 == 0) { // TLDR THIS CONDITION MAKES IT INCORRECT
                				...
                				__syncthreads();
                		}
                		else {
                				...
                				__syncthreads();
                		}
                }
                ```
                
                - in this case, threads with even threadIdx.x values execute one barrier synchronization point, while threads with odd threadIdx.x values execute another barrier synchronization point → so not all threads are guaranteed to execute the first barrier and not all threads are guaranteed to execute the second barrier → this violates the rules and will result in undefined execution behavior
        - incorrect usage of barrier synchronization can result in incorrect results or deadlock (threads waiting for each other forever)
    - barrier synchronization imposes execution constraints on threads within a block:
        1. threads should execute in close time proximity with each other to avoid excessively long waiting times
        2. system needs to make sure that ALL threads involved in the barrier synchronization have access to the necessary resources to eventually arrive at the barrier (otherwise, a thread that never arrives at a barrier causes a deadlock (other threads wait on it forever)
            - **the CUDA runtime satisfies this constraint → (look at figure 4.2) it assigns execution resources to all threads in the block as a UNIT, ensuring that:**
                1. **one block (and all the threads within it) assigned to only one SM**
                2. **and they are assigned simultaneously (so the values of memory locations are the same)**
                - **so a block can only begin execution when the runtime system has secured all the resources needed by all threads in the block to complete execution**
                    - **this ensures time proximity of all threads in a block and prevents excessive or indefinite waiting time during barrier synchronization**
            - for these reasons, CUDA does not allow threads in DIFFERENT blocks to perform barrier synchronization
                - **important TRADEOFF**: **runtime system can now execute blocks in any order relative to each other since none of the blocks wait for each other → this flexibility enables scalable implementations across systems/devices with variable execution resources (eg. in example below, can run on both a system capable of running only 2 blocks at a time and another system capable of running 4 blocks at a time)**
                    
                    ![Screenshot 2024-09-20 at 5.07.49 PM.png](files/Lecture%203%20CUDA%20Parallel%20Execution%20Model/Screenshot_2024-09-20_at_5.07.49_PM.png)
                    
                    - ability to execute the same application code with a wide range of speeds allows the production of a wide range of implementations according to the cost, power, and performance requirements of different market segments
                        - eg. a mobile processor may execute an application slowly but at extremely low power consumption, and a desktop processor may execute the **same** application at a higher speed while consuming
                        more power → important thing to note is that they execute the same application program with no change to the code.
                    - ability to execute the same application code on different hardware with different amounts of execution resources is referred to as **transparent scalability** across different devices, which reduces the burden on application developers and improves the usability of applications

# Resource assignment

- RECALL: when a kernel is called, CUDA runtime system launches a grid of threads that execute the kernel code

## Block scheduling

- RECALL: these threads are assigned to SMs on a block-by-block basis, i.e. all threads in a block are simultaneously assigned to the same SM
    - this guarantees that threads in the same block are scheduled simultaneously on the same SM → this then guarantees that threads in the same block can interact with each other in ways that threads across different blocks can’t, including **barrier synchronization** and **accessing a low-latency shared memory** that resides on the SM
    - multiple blocks are likely to be assigned to the same SM (eg. in the picture, 3 blocks are assigned to each SM) BUT blocks need to reserve hardware resources to execute, so only a limited number of blocks can be simultaneously assigned to one SM (this number depends on the device)
        - TLDR: each device has a limit on the number of blocks that can be assigned to each SM
        
        ![Screenshot 2024-09-20 at 4.01.14 PM.png](files/Lecture%203%20CUDA%20Parallel%20Execution%20Model/Screenshot_2024-09-20_at_4.01.14_PM.png)
        
        - consider a CUDA device that may allow up to 8 blocks to be assigned to each SM → in situations where there is a shortage of one or more types of resources needed for the simultaneous execution of 8 blocks, the CUDA runtime automatically reduces the number of blocks assigned to each SM until their combined resource usage falls below the limit
            - one example of SM resource limitations is the # threads that can be simultaneously tracked and scheduled → this takes hardware resources in the form of built-in registers
- there is a limited number of SMs and a limited number of blocks that can be simultaneously assigned to each SM → so naturally there is a limit on the total number of blocks that can be simultaneously executing in a CUDA device
    - most grids contain many more blocks than this upper limit
- so to ensure that ALL blocks in a grid get executed, the runtime system maintains a list of blocks that still need to execute and then assigns these new blocks to SMs when previously assigned blocks complete execution

# Querying device properties

- exploring how to find out the amount of resources available (in order to understand how execution resources are assigned to blocks)
    - When a CUDA application executes on a system, how can it determine the number of SMs in the device and the number of blocks and threads that can be assigned to each SM
    - since many modern applications are designed to execute on a wide variety of hardware systems, they need to be able to **query** the available resources and capabilities of the underlying hardware in order to take advantage of the more capable systems while compensating for the less capable systems
- **`cudaGetDeviceCount`**: API function provided by the CUDA runtime system (device driver) that returns the number of available CUDA devices in the system
    
    ```c
    // Usage in host code
    int dev_count;
    cudaGetDeviceCount(&dev_count);
    ```
    
    - CUDA runtime numbers all available devices in the system from 0 to `dev_count - 1`
    - While it may not be obvious, a modern PC system often has two or more CUDA devices. This is because many PC systems come with one or more “integrated” GPUs. These GPUs are the default graphics units and provide rudimentary capabilities and hardware resources to perform minimal graphics functionalities for modern window-based user interfaces
        - Most CUDA applications will not perform very well on these integrated devices. This would be a reason for the host code to iterate through all the available devices, query their resources and capabilities, and choose the ones that have enough resources to execute the application with satisfactory performance
- **`cudaGetDeviceProperties`**: API function provided by the CUDA runtime system that returns the properties of the device whose number is given as an argument
    
    ```c
    // Usage in host code to iterate through available devices and query their properties
    cudaDeviceProp dev_prop;
    for (int i = 0; i < dev_count; ++i) {
    		cudaGetDeviceProperties(&dev_prop, i);
    		// code to decide if device has sufficient resources and capabilities
    }
    ```
    
    - `cudaDeviceProp` is a built-in C struct type with fields representing the properties of a CUDA device
        - some notable fields relevant to the assignment of execution resources to threads:
            - `maxThreadsPerBlock`: maximal number of threads allowed in a block
            - `multiProcessorCount`: the number of SMs in the device
            - `clockRate`: the clock frequency of the device
                - combination the clock rate and the number of SMs gives a good indication of the maximum hardware execution throughput of the device
            - `maxThreadsDim[<index>]`: the maximum number of threads allowed along each dimension of a block
                - index = 0 for x-dim, 1 for y-dim, 2 for z-dim
                - an example of use of this information is for an automated tuning system to set the range of block dimensions when evaluating the best performing block dimensions for the underlying hardware.
            - `maxGridSize[<index>]`: the maximum number of blocks allowed along each dimension of a grid
                - index = 0 for x-dim, 1 for y-dim, 2 for z-dim
                - a typical use of this information is to determine whether a grid can have enough threads to handle the entire dataset or some kind of iterative approach is needed
            - `regsPerBlock`: the number of registers that are available in each SM
                - useful in determining whether the kernel can achieve maximum occupancy on a particular device or if it will be limited by its register usage (note the analysis in the previous section)
                - Note that the name of the field is a little misleading → For most compute capability levels, the maximum number of registers that a block can use is indeed the same as the total number of registers that are available in the SM. **However, for some compute capability levels, the maximum number of registers that a block can use is less than the total that are available on the SM.**
            - `warpSize`: the size of warps

# Thread scheduling and latency tolerance

- look in Lecture 4 notes (Warp scheduling and SIMD architecture)
- simple exercise:
    - Assume that a CUDA device allows up to 8 blocks and 1024 threads per SM, whichever becomes a limitation first. Furthermore, it allows up to 512 threads in each block.
        - For image blur, should we use 8 × 8, 16 × 16, or 32 × 32 thread blocks? To answer the question, we can analyze the pros and cons of each choice.
            - If we use 8 × 8 blocks, each block would have only 64 threads. We will need 1024/64 = 12 blocks to fully occupy an SM. However, each SM can only allow up to 8 blocks; thus, we will end up with only 64 × 8 = 512 threads in each SM. This limited number implies that the SM execution resources will likely be underutilized because fewer warps will be available to schedule around long-latency operations.
            - The 16 × 16 blocks result in 256 threads per block, implying that each SM can take 1024/256 = 4 blocks. This number is within the 8-block limitation and is a good configuration as it will allow us a full thread capacity in each SM and a maximal number of warps for scheduling around the long-latency operations.
            - The 32 × 32 blocks would give 1024 threads in each block, which exceeds the 512 threads per block limitation of this device. Only 16 × 16 blocks allow a maximal number of threads assigned to each SM.

# Summary

- CUDA grids and blocks are multidimensional with up to three dimensions → the multidimensionality of grids and blocks is useful for organizing threads to be mapped to multidimensional data
- kernel execution configuration parameters define the dimensions of a grid and its blocks
    - unique coordinates in `blockIdx` and `threadIdx` allow threads of a grid to identify themselves and their domains of data
- when accessing multidimensional data, will have to linearize multidimensional indices into a 1D offset
    - reason is that dynamically allocated multidimensional arrays in C are typically stored as 1D arrays in row-major order
- deciding dimensions for 2D inputs: [https://stackoverflow.com/questions/2392250/understanding-cuda-grid-dimensions-block-dimensions-and-threads-organization-s](https://stackoverflow.com/questions/2392250/understanding-cuda-grid-dimensions-block-dimensions-and-threads-organization-s)
- Once a grid is launched, its blocks can be assigned to SMs in an arbitrary order, resulting in the transparent scalability of CUDA applications. The transparent scalability comes with a limitation: threads in different blocks cannot synchronize with one another. To allow a kernel to maintain transparent scalability, the simple method
for threads in different blocks to synchronize with each other is to terminate the kernel and start a new kernel for the activities after the synchronization point.
- Threads are assigned to SMs for execution on a block-by-block basis. Each CUDA device imposes a potentially different limitation on the amount of resources available in each SM. Each CUDA device sets a limit on the number of blocks and the number of threads each of its SMs can accommodate, whichever becomes a limitation first.
    - For each kernel, one or more of these resource limitations can become the limiting factor for the number of threads that simultaneously reside in a CUDA device.
- Once a block is assigned to an SM, it is further partitioned into warps. All threads in a warp have identical execution timing. At any time, the SM executes instructions of only a small subset of its resident warps. This condition allows the other warps to wait for long-latency operations without slowing d

# Resources

- Kirk & Hwu: Chapter 3 Scalable parallel execution