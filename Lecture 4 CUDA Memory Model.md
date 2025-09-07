# Lecture 4: CUDA Memory Model

# Background

- previous chapter: learned how to write a CUDA kernel function and how to configure and coordinate its execution by a massive number of threads
    - the kernels we have learned so far will likely achieve only a tiny fraction of the potential speed of the underlying hardware → poor performance is attributable to long access latencies (hundreds of clock cycles) and finite access bandwidth of global memory (typically implemented with DRAM)
    - having lots of threads available for execution (see thread scheduling from previous notes) can tolerate long memory access latencies, one can easily run into a situation where traffic congestion in global memory access paths prevents all but few threads from making process → this renders the hardware Streaming Multiprocessors (SMs) idle
    - to circumvent such congestion, CUDA provides a number of additional resources and methods for accessing memory that can remove the majority of traffic to and from the global memory
- this chapter: study how one can organize and position the data for efficient access by a massive number of threads → one part of this includes using different memory types to boost the execution efficiency of CUDA kernels

# Importance of Memory Access Efficiency

Consider this most executed portion (nested for loop) of the image blur kernel code from the previous notes:

```c
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
```

This section illustrates the effect of memory access efficiency by calculating the expected performance level of this portion of code.

- in every iteration of the inner loop, one global memory access (`in[curRow * width + curCol]`) to fetch the input array element is performed for one floating-point addition (`pixVal +=`) that accumulates the value of the fetched element
- **compute-to-global-memory-access-ratio**: $\frac{\text{\# floating-point calculations}}{\text{per access to global memory}}$ within a region of a program → in the example above, the ratio is 1.0
    - this ratio has major implications on the performance of a CUDA kernel:
        - in a high-end device today, the **global memory bandwidth** is around 1000 GB/s → with 4 bytes in each single-precision float, this means that no more than 1000/4 = 250 giga single-precision operations/s can be expected to load
        - with a compute-to-global-memory ration of 1.0, the execution of the image blur kernel will be limited by the rate at which the operands (i.e. the election of the input matrix) can be loaded/delivered to the GPU → these programs whose execution speed is limited by memory access throughput are **memory-bound programs**
            - in the example, the blur kernel will achieve no more than 250 giga floating point operations per second (GFLOPS)
- **in order to achieve a higher level of kernel performance, we need to increase the compute-to-global-memory-access-ratio by reducing the number of global memory accesses**
    - in general, the desired ratio has been increasing in the past few generations of devices because computational throughput has been increasing faster than memory bandwidth
    - this chapter introduces a commonly used technique (tailing) for reducing the number of global memory accesses

# Matrix multiplication

- important component of the Basic Linear Algebra Subprograms standard (a de factor standard for publishing libraries that perform basic algebra operations)
    - 3 levels of linear algebra functions:
        - **Level 1:** functions that perform **vector operations** of the form $\vec{y} = \alpha\vec{x}+\vec{y}$
            - eg. vector addition is an example of a level 1 function with $\alpha=1$
        - **Level 2:** functions that perform **matrix-vector operations** of the form $\vec{y}=\alpha\bold{A}\vec{x}+\beta\vec{y}$, where $\bold{A}$ is a matrix
        - **Level 3:** functions that perform **matrix-matrix operations** of the form $\bold{C}=\alpha\bold{A}\bold{B}+\beta\bold{C}$
            - eg. the matrix multiplication later in these notes is an example of a level 3 function with $\alpha=1$ and $\beta = 0$
    - BLAS functions are important because they are used as basic building blocks of higher-level algebraic functions (eg. linear system solvers and eigenvalue analysis)
- matrix multiplication between an $i \times j$ (i rows by j columns) matrix $M$ and a $j \times k$ matrix $N$ produces an $i \times k$ matrix $P$ where each element of output matrix $P$ is an inner product of a row of $M$ and a column of $N$
    - inner product $P_{row,col}=\sum M_{row,k} * N_{k,col}$ for $k=0,1,...,\text{Width - 1}$ (so you fix row, col and iterate on k)
- matrix multiplication presents opportunities for reduction of global memory accesses with relatively simple techniques
    - the execution speed of matrix multiplication functions can vary by orders of magnitudes depending on the level of reduction of global memory accesses

## Parallelizing matrix multiplication

![Screenshot 2024-09-17 at 3.20.26 PM.png](files/Lecture%204%20CUDA%20Memory%20Model/Screenshot_2024-09-17_at_3.20.26_PM.png)

- map the threads in the grid to the elements of the output matrix $P$ using the same approach as in the vector addition example → i.e. each thread is responsible for calculating one element of $P$
- the thread-to-data mapping effectively divides $P$ into tiles (the lighter color square in the above image) and **each block is responsible for calculating one tile**

```c
// This assumes the kernel only handles square matrices of Width x Width dimensions
__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width) {
		int row = blockIdx.y * blockDim.y + threadIdx.y // recall this determines the y-coord of P
		int col = blockIdx.x * blockDim.x + threadIdx.x // recall this determines the x-coord of P
		
		if ((row < Width) && (col < Width)) {
				float Pvalue = 0;
				// iterating through k with a fixed row and col
				for (int k = 0; k < Width; ++k) {
						Pvalue += M[row * Width + k] * N[k*Width + col];
				}
				
				// each thread then uses the 1D equivalent index mapping to write to P
				// recall row and col are for P and the initialization of row col depends on
				// gridDim and blockDim, which should have already taken into account the necessary
				// dimensions to complete the whole picture for the row*Width + col expression 
				P[row*Width + col] = Pvalue;
		}
}
```

### Accessing elements of input matrices M and N

- $M$ is linearized into an equivalent 1D array using row-major order
    - elements of row 0 are M[0 … Width - 1]
    - elements of row 1 are M[1 * Width … 2*Width - 1]
    - elements of row 2 are M[2 * Width … 3*Width - 1]
    - **generalization: beginning element of row $r$ is M[r * Width]**
    - **since all elements of a row are placed in consecutive locations, the $k^{th}$ element (0-indexed) of the $r^{th}$ row is at M[r*Width + k]**

- $N$ is similarly linearized into an equivalent 1D array in row-major order
    - the **beginning element** of the $c^{th}$ column is the $c^{th}$ element of row 0 → N[c]
    - accessing the next element in the column requires skipping over an entire row (i.e. Width elements) because the next element of the same column is the same element in the next row → **second element (in row 1)** is N[c + 1*Width]
    - **third element (in row 2)** is N[c + Width + Width] = N[c + 2*Width]
    - **generalization: $k^{th}$ element (0-indexed) of the $c^{th}$ column is at N[c + k*Width]**

## Matrix multiplication example

![Screenshot 2024-09-18 at 12.10.02 PM.png](files/Lecture%204%20CUDA%20Memory%20Model/Screenshot_2024-09-18_at_12.10.02_PM.png)

![Screenshot 2024-09-18 at 3.25.23 PM.png](files/Lecture%204%20CUDA%20Memory%20Model/Screenshot_2024-09-18_at_3.25.23_PM.png)

- 4x4 output matrix P with a (BLOCK_WIDTH=2)x2 kernel → this makes P divided into 4/2 = 2 tiles where each block calculates 1 tile
    - so need to create blocks that are 2x2 arrays of threads with each thread calculating one P element
    - in example:
        - thread (0,0) of block (0,0) calculates $P_{0,0}$
        - thread (0,0) of block (1,0) calculates $P_{2,0}$
- `row` and `col` indices in the kernel code identify the element of $P$ to be calculated by a thread
    - `row` also identifies the row of M that is the input for the thread
    - `col` also identifies the col of N that is the input for the thread
    - each are responsible for 4 dot products with 4 accesses to elements of M and 4 accesses to elements of N (because the dimensions of M and N are 4x4)
- walking through the for loop for thread (0,0) in block (0,0):
    - `row = blockIdx.y * blockDim.y + threadIdx.y = 0*2 + 0 = 0`
    - `col = blockIdx.x * blockDim.x + threadIdx.x = 0*2 + 0 = 0`
    - k=0
        - row*Width + k = 0*4 + 0 = 0
        - k*Width + col = 0*4 + 0 = 0
        - so the input elements accessed are M[0] (1D equivalent of $M_{0,0}$) and N[0] (1D equivalent of $N_{0,0}$)
    - k=1
        - row*Width + k = 0*4 + 1 = 1
        - k*Width + col = 1*4 + 0 = 4
        - so the input elements accessed are M[1] (1D equivalent of $M_{0,1}$) and N[4] (1D equivalent of $N_{1,0}$)
    - k=2
        - row*Width + k = 0*4 + 2 = 2
        - k*Width + col = 2*4 + 0 = 8
        - so the input elements accessed are M[2] (1D equivalent of $M_{0,2}$) and N[8] (1D equivalent of $N_{2,0}$)
    - k=3
        - row*Width + k = 0*4 + 3 = 3
        - k*Width + col = 3*4 + 0 = 12
        - so the input elements accessed are M[3] (1D equivalent of $M_{0,3}$) and N[12] (1D equivalent of $N_{3,0}$)

## Other Considerations

- since the size of the grid (array of threads) is limited by the maximum number of blocks per grid and the threads per block, the size of the largest output matrix P that can be handled by this kernel will also be limited by these constraints (naturally)
- if want to use this kernel on larger matrices, can:
    - divide output matrix into submatrices that can be covered by the constraints and have the host code launch the same kernel on different grids for each submatrix
    - or, just modify kernel code so that each thread calculates more than one element of P
- can estimate the effect of memory access efficiency by calculating the expected performance level of the matrix multiplication kernel code
    - the dominating portion is the for loop that performs the inner-product calculations:
        
        ```c
        for (int k = 0; k < Width; ++k) {
        		Pvalue += M[row * Width + k] * N[k*Width + col];
        }
        ```
        
        - in every iteration, 2 global memory accesses (1 fetching from M and 1 fetching  from N) are performed for 2 floating-point operations (1 floating-point multiplication and 1 floating point addition) → this means that the compute-to-global-memory-access ratio of this for loop is 1.0
            - From our discussion in Chapter 3, Scalable parallel execution, this ratio will likely result in less than 2% utilization of the peak execution speed of the modern GPUs → we need to increase the ratio by at least an order of magnitude for the computation throughput of modern devices to achieve good utilization
                - in the next section, we will show that we can use special memory types in CUDA devices to accomplish this goal.

# CUDA Memory Types

- a CUDA device contains several types of memory → programmers can declare a CUDA variable in one of these memory types, and as a result dictate the visibility and access speed of the variable
    
    ![Screenshot 2024-09-30 at 4.12.54 PM.png](files/Lecture%204%20CUDA%20Memory%20Model/Screenshot_2024-09-30_at_4.12.54_PM.png)
    
    - **global memory:** introduced in Ch. 2
        - host: can write (W) and read (R) via API
        - device: can write (W) and read (R)
    - **constant memory:** supports short-latency, high-bandwidth read-only access by the device
        - host: can write (W) and read (R) via API
        - device: can only READ
    - **registers:** on-chip memory
        - variables in registers can be accessed at a very high-speed in a highly parallel manner
        - each INDIVIDUAL thread is allocated a set of registers → each thread can only access its own registers
        - kernel functions typically use registers to hold frequently accessed variables that are private to each thread
    - **shared memory:** on-chip memory
        - variables in shared memory can be accessed at a very high-speed in a highly parallel manner
        - each thread BLOCK is allocated some shared memory → all threads in the block can access shared memory variables allocated to the block
        - an efficient means for threads to cooperate by sharing their input data and intermediate results

## How different memory types are realized and used in modern processors

- this will highlight the differences between registers, shared memory, and global memory
- virtually all modern processors (including CUDA devices) are rooted in the model proposed by John von Neumann in 1945
    
    ![Screenshot 2024-09-30 at 4.48.48 PM.png](files/Lecture%204%20CUDA%20Memory%20Model/Screenshot_2024-09-30_at_4.48.48_PM.png)
    
    - in CUDA devices:
        - processor chip boundary corresponds to the (entire) Processor box
        - **global memory** corresponds to the Memory box → **notice how it is OFF the processor chip**
            - implemented with DRAM technology → this implies long access latencies and relatively low access bandwidths
        - **registers** correspond to the Register file box → **notice how it is ON the processor chip**, which implies very short access latency and drastically higher bandwidth compared to global memory
            - the aggregated access bandwidth of register files is at least 2 orders of magnitude higher than that of the global memory
            - when a variable is stored in a register, its accesses no longer consume off-chip global memory → good as the reduced bandwidth consumption is reflected as an increased comput-to-global-memory-access ratio
            - **subtle advantage 1**: each access to a register involves fewer instructions than each access to the global memory because arithmetic instructions in most modern processors have “built-in” operand registers
                - eg. a floating-point addition instruction might be of the form `fadd r1, r2, r3` where `r2` and `r3` are the register numbers that specify the location in the register file where the input values (the operands) can be found, and `r1` is the location where the resulting value will be stored → this means that **after an operand of an arithmetic instruction is in a register, no additional instruction is required to make the operand values available to the ALU where the calculation is performed**
                - in contrast: if an operand value is in the global memory, the processor needs to perform a memory load operation to make it available to the ALU
                    - eg. if the operand of a floating-point addition instruction is in the global memory, there would be an additional `load r2, r4, offset` instruction that adds an offset value to the contents of `r4` to form an address for the operand value to be in `r2` , before the usual `fadd r1, r2, r3` instruction
                    - the processor can only fetch and execute a limited number of instructions per clock cycle, so the additional load will incur more execution time
                        - so placing the operands in registers can improve execution speed
            - **subtle advantage 2 when placing an operand value in registers**: in modern computers, the energy consumed for accessing a value from the register file is at least an order of magnitude lower than that for accessing a value from the global memory
            - don’t forget that the number of registers available to each thread is limited, so need to be careful not to oversubscribe to this limited resource

### Shared Memory vs. Registers

![Screenshot 2024-09-30 at 8.49.46 PM.png](files/Lecture%204%20CUDA%20Memory%20Model/Screenshot_2024-09-30_at_8.49.46_PM.png)

- although both are on-chip memories, they differ significantly in functionality and cost of access
- **shared memory** is designed as part of the memory space that resides on the processor ship
    - when a processor accesses data in shared memory, it needs to perform a **memory load operation** (similar to accessing data in the global memory) → but because shared memory is *on-chip*, it can be accessed with much lower latency and much higher throughput
    - shared memory has longer latency and lower bandwidth than registers because of this need for a memory load operation
    - shared memory is also known as *scratchpad memory* in computer architecture terminology
- **important difference:** variables in shared memory are accessible by ALL THREADS IN A BLOCK, whereas data in registers are PRIVATE TO A THREAD (this was also noted above)
    - shared memory is designed to support efficient, high-bandwidth sharing of data among threads in a block → if you look at Figure 4.8 above, you can see that a CUDA device SM has multiple Processing units, which allows multiple threads to simultaneously execute instructions → since threads in a block can be spread across multiple of these processing units, the hardware implementations of the shared memory are designed to allow multiple processing units to simultaneously access its contents to support efficient data sharing among threads in a block

## Aside: von Neumann Model

![Screenshot 2024-09-21 at 11.47.12 AM.png](files/Lecture%204%20CUDA%20Memory%20Model/Screenshot_2024-09-21_at_11.47.12_AM.png)

- blueprint model for all modern computers based on the design of the EDVAC computer
- **I/O module**: allows both programs and data to be provided to and generated from the system
- to execute a program, the computer first inputs the program and data into the **Memory module**
- a program consists of a collection of instructions
    - the **Control Unit** maintains a **program counter (PC)** which contains the memory address of the next instruction to be executed
    - in each instruction cycle, the Control Unit uses the PC to fetch and instruction into the **Instruction Register (IR)**
    - instruction bits are examined to determine the action to be taken by all components of the computer → this is why the model is also called the “stored program” model because a user can change the behavior of a computer by storing a different program into its memory

## Aside: Processing Units and Threads, Context-switching, and Zero-overhead Scheduling

- getting a deeper understanding of how threads are implemented
- a thread in modern computers is (1) a program and (2) the state of the executing the program on a von Neumann Processor → recall that it (the thread) consists of the program code, the instruction in the code that is being executed, and the value of its variables and data structures
- in a computer based on the von Neumann model:
    - the code of the program is stored in memory
    - the PC (program counter) keeps track of the *address* of the current instruction in the program being executed
    - the IR (instruction register) holds the actual instruction that is being executed
    - the register and memory hold the values of the variables and data structures
- modern processors are designed to allow **context-switching:** concept where multiple threads can time-share a processor by taking turns to make progress.
    - able to do this (i.e. suspend the execution of a thread and correctly resume the execution of the thread later) by carefully saving and restoring the PC value and the contents of registers and memory
        - however, saving and restoring register contents during context-switching in these processors can incur significant overhead in terms of added execution time.
    - some processors provide multiple processing units, which allow multiple threads to make simultaneous progress → Figure 4.8 boave shows a single-instruction, multiple-data (SIMD) design style with multiple processing units sharing a PC and IR → in this design, all threads make simultaneous progress by executing the same instruction in the program (SIMD will be covered later)
- **Zero-overhead scheduling:** refers to the GPU’s ability to put a warp that needs to wait for a long-latency instruction result to sleep and activate a warp that is ready to go without introducing any extra idle cycles in the processing units
    - Traditional CPUs incur such idle cycles because switching the execution from one thread to another requires saving the execution state (such as register contents of the out-going thread) to memory and
    loading the execution state of the incoming thread from memory.
    - GPU SMs achieves zero-overhead scheduling by holding all the execution states for the assigned warps in the hardware registers so there is no need to save and restore states when switching from one warp to another.

## Declaring variables of different CUDA memory types

- RECAP: registers, shared memory, and global memory all have different functionalities, latencies, and bandwidths
- as such, need to specify where a variable will reside when declaring it
    
    ![Syntax for declaring program variables into various memory types](files/Lecture%204%20CUDA%20Memory%20Model/Screenshot_2024-09-30_at_9.29.32_PM.png)
    
    Syntax for declaring program variables into various memory types
    
    - **scope**: the range of threads that can access the variable → can be:
        - **single thread only** → in which case a private version of the variable will be created for every thread, and each thread can only access its private version of the variable
        - **all threads of a block**, or
        - **all threads of all grids**
    - **lifetime**: the portion of the program execution duration when the variable is available for use → can be:
        - **within a kernel execution** → in which case it must be declared within the kernel function body and will be available for use only by the kernel code
            - if the kernel is invoked several times, the value of variable is NOT maintained across different invocations
            - each invocation must initialize the variable in order to use → a private version of the shared variable is created for and used by each thread block during kernel execution
        - or **throughout the entire application** → in which case it must be declared outside of any function body
            - the contents of these variables are maintained throughout the execution of the application and available to all kernels

### Scalar Variables

- **scalar** variables: any variables that are not arrays or matrices
    - as seen from table 4.1: all ***automatic* scalar** variables declared in kernel and device functions are placed into **registers**, scoped to just individual threads
        - scope: as with any other thread-scoped variable, when a kernel function declares an automatic variable, **a private copy is generated for every thread that executes the kernel function**
        - lifetime: when a thread terminates (i.e. after executing the kernel code), all its automatic variables cease to exist
        - eg. in the image blur kernel code from before, `blurRow`, `blurCol`, `curRow`, `curCol`, `pixels`, and `pixVal` are automatic scalar variables
        - again, don’t forget that the number of registers available to each thread is limited, so need to be careful not to oversubscribe to this limited resource
            - using a large number of registers can also negatively affect the number of active threads assigned to each SM

### Automatic Array Variables

- **automatic array variables**: stored into **global memory**, and not in registers → thus, may incur long access delays and potential access congestions
    - scope: (same as automatic scalar variables) scoped to just individual threads
        - **a private version of each automatic array is created for and used by every thread**
    - lifetime: when a thread terminates (i.e. after executing the kernel code), the contents of its automatic array variables also cease to exist

### `__s**hared__` keyword**

- declares a shared variable located in shared memory, and an
- declared typically  within a kernel function or a device function, and an optional `__device__` keyword can be added in declaration
- scope: within a thread block, so all threads in a block see the same version of the shared variable
    - **a private version is created per thread block during kernel execution**
- lifetime: when the kernel terminates execution, the contents of its shared variables cease to exist
- often used to hold the portion of global memory data that are heavily used in a kernel execution phase

### `__constant**__` keyword**

- declares a constant variable, which is stored in constant memory per the table 4.1 (global memory but **cached for efficient access** (I think this is the definition of constant memory))
- must be declared OUTSIDE of any function body, and an optional `__device__` keyword can be added in declaration
- scope: all grids, so all threads in all grids see the same version of a constant variable
- lifetime: the entire application execution
- often used for variables that provide input values to kernel function
    - currently, the total size of constant variables in an application is limited to 65,536 bytes, so input data volume may need to be divided to fit within this limitation
    - with appropriate access patterns, accessing constant memory is extremely fast and parallel

### `__**device__` keyword only**

- declares a global variable stored in global memory → thus, may incur long access delays and potential access congestions (thought latency and throughput have been improved with caches in relatively recent devices)
- scope: all grids, so all threads in all grids see the same version of a global variable
    - this is an important advantage (visible to all threads of all kernels)
- lifetime: entire application execution
- so global variables can be used as a means for threads to collaborate across blocks
    - however, the only easy way to synchronize between threads from different thread blocks or to ensure data consistency across threads when accessing global memory is by terminating the current kernel execution
    - so in reality, global variables are often used to pass information from one kernel invocation to anther kernel invocation

### Pointers

- in CUDA, pointers are used to point to data objects in the global memory
- usage arises in kernel and device functions in two ways:
    1. if an object is allocated by a host function, the pointer to the object is initialized by `cudaMalloc` (recall first lecture) and can be passed to the kernel function as a parameters
    2. the address of a variable declared in the global memory is assigned to a pointer variable
        - eg. the statement `float* ptr= &GlobalVar;` in a kernel function assigns the address of `GlobalVar` into an automatic pointer variable `ptr`
- refer to the CUDA Programming Guide for using pointers in other memory types

# Tiling for Reduced Memory Traffic

- RECAP: there is an intrinsic tradeoff when using different types of CUDA device memories → global memory is LARGE but SLOW, whereas shared memory is SMALL but FAST
- to optimize this, a common strategy is to partition data into subsets called **tiles** so that each tile fits into the shared memory
    - analogy: a large wall (i.e. global memory data) can be covered by tiles (i.e. subsets that can each fit into the shared memory)
    - **important criterion:** kernel computation on these tiles can be performed independently of each other (so given an arbitrary kernel function, not all data structures can be partitioned into tiles for the kernel function

## Matrix Multiplication Example

![Screenshot 2024-10-01 at 3.56.26 PM.png](files/Lecture%204%20CUDA%20Memory%20Model/Screenshot_2024-10-01_at_3.56.26_PM.png)

- each thread (working on one element of P) accesses 4 elements of M and 4 elements of N
    - but notice the redundant global memory accesses done by all threads in this block $_{0,0}$ (the red and blue boxed accesses, for example)
        - consider thread$_{0,0}$ and thread$_{0,1}$ working on $P_{0,0}$ and $P_{0,1}$ respectively:
            - naive kernel: both threads access row 0 of M
            - smarter kernel: make both threads collaborate so that row 0 of M is only loaded from global memory once → this reduces the total number of accesses to global memory by 1/**2** because every M and N element is accessed exactly **2x** during the execution of block$_{0,0}$ because it is a **2x2** block
        - so if all 4 threads can be made to collaborate in their accesses to global memory, traffic to global memory can be halved
            - the potential reduction in global memory traffic in the matrix multiplication example is proportional to the dimensions of the block used → with NxN blocks, the potential reduction would be 1/N
- analogy: consider threads as commuters and DRAM (global memory) access requests as vehicles → when the rate of DRAM requests exceeds the provisioned access bandwidth of the DRAM system, traffic congestion arises and the arithmetic units become idle
    - if multiple threads need to access data from the same DRAM location, they can carpool and combine accesses into one DRAM request (vehicle)
    - core requirement for this is that the carpooling threads need to have similar enough execution schedules to make sense
        
        ![Deep blue cells represent DRAM locations; arrow from the cell to the thread represents an access by that thread to that location at the time marked by the head of the arrow](files/Lecture%204%20CUDA%20Memory%20Model/Screenshot_2024-10-01_at_4.13.03_PM.png)
        
        Deep blue cells represent DRAM locations; arrow from the cell to the thread represents an access by that thread to that location at the time marked by the head of the arrow
        
        - having very different timing is an undesirable arrangement because the **data elements that are brought back from the DRAM need to be stored in the on-chip memory for an extended time to wait to be consumed by the lagging thread (thread 2 in the above example)** → if this arrangement is not controlled, then a large number of data elements will need to be stored, resulting in an excessive on-chip memory requirement
- **tiling:** essentially a program transformation technique that localizes the memory locations accessed (space) among threads and localizes the timing of their accesses (time)
    - divides long access sequences of each thread into phases, and uses barrier synchronization to keep the timing of access phases at close intervals
    - as a result, it controls the amount of on-chip memory required
    - in terms of the carpool analogy, we are essentially forcing the threads that can carpool to follow approximately the same execution timing (with the synchronization)

# A Tiled Matrix Multiplication Algorithm/Kernel

## Algorithm

- basic idea: threads collaboratively load subsets of $M$ and $N$ elements into shared memory and compute portions of the dot product in phases given what has been loaded
    - criteria: size of shared memory is quite small, so need to ensure the capacity of the shared memory is not exceeded when these $M$ and $N$ elements are loaded in
        - this can be satisfied by dividing the $M$ and $N$ matrices into smaller tiles that ft into shard memory → in the simplest form, the tile dimension = block dimension
            - again, the reduction in the number of accesses to the global memory occurs by a factor of N if the tiles are N × N elements
    
    ![M and N are divided by the think lines into 2x2 tiles](files/Lecture%204%20CUDA%20Memory%20Model/Screenshot_2024-10-01_at_4.31.47_PM.png)
    
    M and N are divided by the think lines into 2x2 tiles
    
- in each execution phase, all 4 threads in the 2x2 block each load in 1 element from $M$ and 1 element from $N$ into shared memory, collectively loading in a tile of $M$ (4 elements) and a tile of $N$ (4 elements)
    
    ![Mds and Nds are the shared memory arrays. The PValue calculations in Phase 2 (1-indexed) are wrong, look below instead.](files/Lecture%204%20CUDA%20Memory%20Model/Screenshot_2024-10-01_at_4.46.04_PM.png)
    
    Mds and Nds are the shared memory arrays. The PValue calculations in Phase 2 (1-indexed) are wrong, look below instead.
    
    - in Phase 1 (first phase, 1-indexed):
        - thread$_{0,0}$ loads in $M_{0,0}$ and $N_{0,0}$ → $P_{0,0}$ is partially calculated with $M_{0,0} * N_{0,0}$ and $M_{0,1} * N_{1,0}$
        - thread$_{0,1}$ loads in $M_{0,1}$ and $N_{0,1}$ → $P_{0,1}$ is partially calculated with $M_{0,0} * N_{0,1}$ and $M_{0,1} * N_{1,1}$
        - thread$_{1,0}$ loads in $M_{1,0}$ and $N_{1,0}$ → $P_{1,0}$ is partially calculated with $M_{1,0} * N_{0,0}$ and $M_{1,1} * N_{1,0}$
        - thread$_{1,1}$ loads in $M_{1,1}$ and $N_{1,1}$ → $P_{1,1}$ is partially calculated with $M_{1,0} * N_{0,1}$ and $M_{1,1} * N_{1,1}$
    - in Phase 2 (second phase, 1-indexed):
        - thread$_{0,0}$ loads in $M_{0,2}$ and $N_{2,0}$ → $P_{0,0}$ is fully calculated after adding $M_{0,2} * N_{2,0}$ and $M_{0,3} * N_{3,0}$
        - thread$_{0,1}$ loads in $M_{0,3}$ and $N_{2,1}$ → $P_{0,1}$ is fully calculated after adding $M_{0,2} * N_{2,1}$ and $M_{0,3} * N_{3,1}$
        - thread$_{1,0}$ loads in $M_{1,2}$ and $N_{3,0}$ → $P_{1,0}$ is fully calculated after adding $M_{1,2} * N_{2,0}$ and $M_{1,3} * N_{3,0}$
        - thread$_{1,1}$ loads in $M_{1,3}$ and $N_{3,1}$ → $P_{1,1}$ is fully calculated after adding $M_{1,2} * N_{2,1}$ and $M_{1,3} * N_{3,1}$
    - some things to note:
        - each value in the shared memory is used 2x in dot product calculations
        - the variable `Pvalue` where the products are accumulated is an automatic scalar variable, so a private version is generated fro each thread
        - as shared variables, the scope of `Mds` and `Nds` is the block, so they are reused in each phase to hold input values → this allows a much smaller shared memory to be utilized with the locality (i.e. focused access behavior by focusing on a small subset of the input matrix elements)
            - when an algorithm exhibits locality, the opportunity arises to use small, high-speed memories in order to serve most of the accesses and remove these accesses from the global memory → locality is as important for achieving high-performance in multi-core CPUs as in many-thread GPUs (more in Ch. 5)
- in general (for square matrices), if an input matrix is of the dimension `Width x Width` and the tile size is referred to as `TILE_WIDTH` (so the tile dimension is `TILE_WIDTH x TILE_WIDTH`), the dot product would be performed in `Width/TILE_WIDTH` phases (naturally, this is just how many tiles are needed to fit into the entire input matrix, need to do ceiling checks and width AND height checks for non-square input matrices and matrices with dimensions that are not a multiple of the tile width)
    - these phases are key to the reduction of global memory accesses → because each phase focuses on a small subset of the input matrix values, the threads can collaboratively load the subset into the shared memory to satisfy overlapping input demands in the phase

## Kernel

```c
// This assumes the kernel only handles square matrices of Width x Width dimensions
__global__ void MatrixMulKernelTiled(float * M_d, float * N_d, float * P_d, int Width) {
		// Declare **shared memory** variables
		// Recall: the scope of these will be a block, so one pair of Mds, Nds will be created for each block
		// and all threads of a block can access the same Mds and Nds -> this is important because all threads
		// must have access to M and N elements loaded by their thread peers (in each phase, parts of Mds, Nds will be filled)
		__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
		__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
		
		// Save threadIdx and blockIdx values into automatic scalar variables (so into **registers** for fast access)
		int bx = blockIdx.x;
		int by = blockIdx.y;
		
		int tx = threadIdx.x;
		int ty = threadIdx.y;
		
		// Identify the row and column of the P_d element to work on
		// (SIMILAR to before, using TILE_WIDTH because each block covers TILE_WIDTH elements in each dimension)
		// Again, this assumes that each thread is responsible for calculating one P_d element
		int row = by * TILE_WIDTH + ty;
		int col = bx * TILE_WIDTH + tx;
		
		
		float PValue = 0;
		
		// Loop over all the M_d and N_d tile pairs (i.e. the number of phases) required to cover the entire input matrices
		// Recall: each phase uses 1 tile of M and 1 tile of N, so at the beginning of each phase,
		// ph*TILE_WIDTH pairs of M and N elements have been processed by previous phases.
		
		for (int phase = 0; phase < Width/TILE_WIDTH; ++phase) {		// LINE 8
				// Collaborative loading of M_d and N_d tiles into shared memory
				Mds[ty][tx] = M_d[(row * Width) + (phase * TILE_WIDTH) + tx];
				Nds[ty][tx] = N_d[(ph * TILE_WIDTH + ty) * Width + Col];
				
				// This ensures that all threads have finished loading the tiles 
				// before any of them can move forward
				__syncthreads();
				
				// Performs one phase of the dot product
				// (partial dot product contributing to WIDTH/TILE_WIDTH of the necessary computations)
				for (int k = 0; k < TILE_WIDTH; ++k) {
						Pvalue += Mds[ty][k] * Nds[k][tx];
				}
				
				// This ensures that all threads have finished using elements in the Mds and Nd sshared memory
				// before any of them can move on to the next iterations and load different elements in.
				// This prevents threads from loading in different values too early and corrupting the input values
				// for other threads still on the previous iteration
				__syncthreads(); // LINE 14
		}
		
		P_d[row*WIDTH + col] = Pvalue;
}
```

### Writing to shared memory matrices Mds and Nds

![Blue tiles are part of one phase, orange tiles are part of another phase.](files/Lecture%204%20CUDA%20Memory%20Model/Screenshot_2024-10-01_at_6.12.46_PM.png)

Blue tiles are part of one phase, orange tiles are part of another phase.

- by the time we write to $Mds$ and $Nds$, we already know the row of $M$ (`row`) and column of $N$ (`col`) to be processed by the threads

- each block has `TILE_WIDTH * TILE_WIDTH = TILE_WIDTH$^2$` threads (the yellow square) that will collaborate to load `TILE_WIDTH * TILE_WIDTH = TILE_WIDTH$^2$` elements of $M$ into $Mds$
    - because there can be a 1:1 mapping because the numbers are the same, we only need to assign each thread to load one element of $M$
- the beginning column index of the section of $M$ to be loaded is `phase * TILE_WIDTH` → so an easy approach is to have every thread load an element that is `tx` positions away from that beginning point
    - **so the column value is retrieved by `phase * TILE_WIDTH + tx`**
- **the row that the pixel would have to calculate is just equal to the row in $M$, which can be retrieved by skipping forward the `Width of M` by a factor of the `row` index (this factors in the linearization of $M$)**
    
    ![Screenshot 2024-10-02 at 12.18.10 PM.png](files/Lecture%204%20CUDA%20Memory%20Model/Screenshot_2024-10-02_at_12.18.10_PM.png)
    
- as before, form the linearized index using the row*width + col format:
    - **since our row = `row`and the column = `phase * TILE_WIDTH + tx`, the thread needs to load in is `(row*Width) + (phase * TILE_WIDTH + tx)`**
- since `row` is a linear function of `ty`, each of the threads will load a unique element into shared memory
- then just need to write to that thread’s location in the shared memory (that has dimensions `TILE_WIDTH * TILE_WIDTH`) at [ty][tx] because the threadIdxs are for each thread in the block and the size of the block is the same as the size of the tile

- again, assign each thread to load one element of $N$
- column: column index is the same as the thread column (like row index was the same for thread row when getting elements from $M$)
- row: similar to the case with $M$ but imagine the tiled is rotated tile 90 degrees (so instead of considering the yellow bar going right, consider it going down)
    - the beginning row index of the section of $N$ to be loaded is `phase * TILE_WIDTH` → so an easy approach is to have every thread load an element that is `ty`  (because the size of shared memory will be the size of the tile which is the size of the block which uses tx and ty) positions away from that beginning point
- as before, form the linearized index using the row*width + col format:
    - **since our row = `phase * TILE_WIDTH + ty` and the column = `col`, the thread needs to load in is `(phase * TILE_WIDTH + ty) * Width + col`**

### Strip Mining

- lines 8 to 14 illustrate the technique of **strip mining**: taking a long-running loop and breaking it into phases that each consist of an inner loop executing a number of consecutive iterations of the original loop
- the original long-running loop becomes an outer loop that serves to iteratively invoke the inner loop so that all iterations of the original loop are executed in the original order
    - the barrier synchronizations before and after the inner loop forces all threads in the same block to focus their work entirely on a section of their input data before the outer loop moves on to another iteration next in the order

## Benefits of the Tiled Algorithm

- for matrix multiplication, the global memory accesses are reduced by a factor of `TILE_WIDTH`
    - eg. if we use 16 x 16 tiles, can reduce the global memory accesses by a factor of 16 → this increases the compute-to-global-memory-access ratio from 1 to 16
        - this allows the memory bandwidth of a CUDA device to support a computation rate closer to peak performance, since eg. a device with 150 GB/s global memory bandwidth can approach (150 GB/s / 4 B per float) = 37.5 giga floats/s, then 37.5 giga floats/s * 16 operations = 600 GFLOPS

## WARNING: Assumptions were Made for this Kernel

1. the width of the matrices was assumed to be a multiple of the width of the thread blocks → so this kernel can’t correctly process arbitrary-sized matrices
2. the matrices are square matrices

- the boundary checking in the next section walks through a kernel that removes the need for these assumptions

# Boundary Checks

- this section extends the `MatrixMulKernelTiled` kernel above to handle matrices with arbitrary widths
    - width is now not necessarily a multiple of the tile width
    - but this is still assuming square input matrices
- consider $M$ and $N$ to be 3x3 matrices now, making $P$ als oa 3x3 matrix
    - the tile is still 2x2, so `Width = 3` is not a multiple of the `TILE_WIDTH = 2`
    
    ![This shows the memory access pattern during phase 1 (NOTE THAT THIS IS THE SECOND PHASE BECAUSE THE PHASES ARE 0-INDEXED) of block_0,0](files/Lecture%204%20CUDA%20Memory%20Model/Screenshot_2024-10-02_at_4.14.55_PM.png)
    
    This shows the memory access pattern during phase 1 (NOTE THAT THIS IS THE SECOND PHASE BECAUSE THE PHASES ARE 0-INDEXED) of block_0,0
    
    - thread$_{0,1}$ (for the cell right of $P_{0,2}$) and thread$_{1,1}$ (for the cell right of $P_{1,2}$) will attempt to load the nonexistent $M$ elements
    - thread$_{1,0}$ (for $P_{1,2}$) and thread$_{1,1}$ (for the cell right of $P_{1,2}$) will attempt to load the nonexistent $N$ elements
    - loading nonexistent elements is problematic because the system will either will return incorrect elements or abort the program:
        1. accessing elements **past the end of a row**
            - eg. $M$ accesses by thread$_{0,1}$ and thread$_{1,1}$ in Figure 4.18 above will attempt to access $M_{0,3}$ and $M_{1,3}$ which both don’t exist → **what will they access instead? To answer that, consider the linearized layout of 2D matrices:**
                - the element after $M_{0,2}$ in the linearized layout of a 3x3 matrix is $M_{1,0}$, so trying to access $M_{0,3}$ will actually give you $M_{1,0}$
        2. accessing elements **past the end of a column** → again, will return incorrect elements or abort the program
            1. eg. $N$ accesses by thread$_{1,0}$ and thread$_{1,1}$ in Figure 4.18 above will attempt to access $N_{3,0}$ and $N_{3,1}$ which both don’t exist
                - these accesses are to memory locations outside the allocated area for the array
- **it is important to note that these problematic accesses can occur in any phase, and not just the last phase with the edge cases**
    - eg. this is an example of the memory access pattern of block$_{1,1}$ during phase 0 (the first phase) (recall block 1 is the bottom right 2x2 grid that should account for $P_{2,2}$)→ thread$_{1,0}$ and thread$_{1,1}$ attempt to access nonexisting elements $M_{3,0}$ and $M_{3,1}$ and thread$_{0,1}$ and thread$_{1,1}$ attempt to access nonexisting elements $N_{0,3}$ and $N_{1,3}$
        
        ![THIS IS THE MEMORY ACCESS PATTERN FOR BLOCK_{1,1} IN PHASE 0, NOT BLOCK_{1,0}](files/Lecture%204%20CUDA%20Memory%20Model/Screenshot_2024-10-02_at_10.14.40_PM.png)
        
        THIS IS THE MEMORY ACCESS PATTERN FOR BLOCK_{1,1} IN PHASE 0, NOT BLOCK_{1,0}
        
- **it is also important to note that these problematic accesses can NOT be prevented by excluding the boundary threads (empty cells) that calculate invalid $P$ elements because they need to do work for other elements in the block**
    - eg. thread$_{1,0}$ in block$_{1,1}$ doesn’t calculate any valid $P$ element (the cell below $P_{2,2}$ is not valid), but it DOES need to load $M_{2,1}$ during phase 0 for other threads in block$_{1,1}$ to use
- **it is also important to note that some threads that calculate valid $P$ elements will still attempt to access $M$ or $N$ elements that don’t exist**
    - eg. thread$_{0,1}$ in block$_{0,0}$ calculates a valid $P_{0,1}$ element in phase 0, but in phase 1, it attempts to access an invalid $M_{0,3}$ (think about this and make sure you understand it)
- **TLDR:** different boundary condition tests need to be conducted for loading $M$ tiles, $N$ tiles, and calculating/storing $P$ elements

## Boundary test conditions for loading input tiles

- need to test input element for validity → look if `y` index is within number of rows and if  `x` index is within the number of cols
    - ORIGINAL: `Mds[ty][tx] = d_M[(Row)*Width + (ph*TILE_WIDTH + tx)];`
        - NEW: check `Row < Width` and `ph*TILE_WIDTH + tx < Width` before loading from $M$
    - ORIGINAL:  `Nds[ty][tx] = d_N[(ph*TILE_WIDTH + ty)*Width + Col];`
        - NEW: check `ph*TILE_WIDTH + ty < Width` and `Col < Width` before loading from $N$
- if these conditions are not satisfied, should not load the element → instead, just place a value of 0.0 into shared memory because then 0.0 + x = x, which means that the 0.0 won’t affect the calculation of the inner product

## Boundary test conditions for calculating/storing output tile

- thread should only store its final inner/dot product value if it is responsible for a valid $P$ element
    - NEW: check `(Row < Width)` and `(Col < Width)`

## Tiled Matrix Multiplication Kernel with Boundary Condition Checks (main loop only)

- again, this assumes square `Width x Width` matrices only

```c
	 	for (int phase = 0; phase < **ceil(Width/(float) TILE_WIDTH)**; ++phase) {		// LINE 8
				
				**if ((row < Width) && ((phase * TILE_WIDTH + tx) < Width)) {**
						Mds[ty][tx] = M_d[(row * Width) + (phase * TILE_WIDTH) + tx];
				**}
				
				if (((ph * TILE_WIDTH + ty) < Width) && (col < Width)) {**
						Nds[ty][tx] = N_d[(ph * TILE_WIDTH + ty) * Width + Col];
				**}**
				
				__syncthreads();
				
				for (int k = 0; k < TILE_WIDTH; ++k) {
						Pvalue += Mds[ty][k] * Nds[k][tx];
				}
				__syncthreads(); // LINE 14
		}
		
		**if (row < Width && col < Width) {**
				P_d[row*WIDTH + col] = Pvalue;
		**}**
```

## Generalizing to non-square matrices

1. Replace `Width` argument with specific notions of `height of $M$ = height of $P$`, `width of $M$ = height of $N$`, and `width of $N$ = width of $P$` 

# Memory as a Limiting Factor to Parallelism

- RECAP: leveraging CUDA registers and shared memory can be very effective in reducing the number of global memory accesses, but **one needs to be careful about staying within the capacities of these memories**
    - this is because these memories are resources that are necessary for thread execution, but each CUDA device only offers a limited amount of these resources, so we have a limit on the number of threads that can simultaneously reside in the SM for any given application
        - the more resources each thread requires, the fewer the number of threads that can reside in the SM, which means the fewer number of threads that can run in parallel in the entire device
- the usage of these two memory types place limits on the number of threads assigned to each SM

## Register usage vs. level of parallelism

- assume that in a device D, each SM can accommodate up to 1536 threads and 16384 registers:
    - so, to support 1536 threads, each thread can only use 16384/1536 = 10.667 registers → 10 registers per thread
    - if each thread uses 11 registers, then the number of threads that can be executed concurrently in each SM will be reduced → this reduction occurs at the block granularity (i.e. if each block contains 512 threads, then the reduction of threads will be accomplished by reducing 512 threads at a time → the next smaller number of threads is 1536 - 512 = 1024, which is a 1- (1024/1536) = 1/3 reduction in the number of threads simultaneously running in each SM)
- reducing the number of threads that can simultaneously run in each SM can substantially reduce the number of warps available for scheduling, which in turn decreases the ability of the the processor to find useful warps to run when certain warps exhibit long-latency operations
- the number of registers available to each SM varies from one device to another → use the `cudaGetDeviceProperties` function to return `device.regsPerBlock` to figure out how many registers can be used in the kernel for each thread in the SM

## Shared memory usage vs. level of parallelism

- assume the same device D has 16,384 (16K) bytes of shared memory per SM that can be allocated to thread blocks and assume that each SM in D can accommodate up to 8 blocks
    - so to reach this maximum of 16K bytes, each block can’t use more than 2K bytes of shared memory → otherwise, the number of blocks that can be accommodated in each SM is reduced such that the total amount of shared memory used does not exceed 16K bytes
        - eg. if each block uses 5K bytes, then no more than 3 blocks (15K bytes) can be assigned to each SM
- for the float matrix multiplication example, shared memory can become a limiting factor
    - for TILE_WIDTH = 16, each block needs (16 x 16) elements * (4 floats/element) = 1024 = 1K bytes of storage for Mds and Nds each → so each block uses 2KB of shared memory
        - then, the 16KB shared memory allows for 8 blocks to simultaneously operate in an SM → thus, shared memory is not a limiting factor for this tile size as 16KB is the maximum supported by the threading hardware BUT different tile sizes have different analyses
            - in this case, the real limitation is the threading hardware limitation that only allows 1536 threads in each SM → this limits the number of blocks in each SM to 6 because each block is now configured (in our kernel example, i think because the block size doesn’t necessarily have to be the tile size) to use 16 x 16 = 256 threads and 1536 max threads in SM / (256 threads per block) = 6 blocks
                - so only 6 * 2KB = 12KB of shared memory will be used
- again, the size of shared memory in each SM varies from one device to another → use the `cudaGetDeviceProperties` function to return `device.sharedMemPerBlock` to figure out how much shared memory can be used in the kernel and determine the amount of shared memory that should be used by each block (recall scope of shared memory is a block) and adjust algorithm dynamically
    - the kernel above doesn’t support this: when declaring Mds and Nds like `__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];`, it hardcodes the size of the shared memory usage to the compile-time constant `#define TILE_WIDTH 16` → so the size of Mds and Nds are always `TILE_WIDTH * TILE_WIDTH` elements regardless of the value of `TILE_WIDTH` at compile-time
        - if want to adjust shared memory usage, need to change constant and recompile
    - to allow for adjustment of the Mds and Nds sizes dynamically:
        - add C `extern` keyword in front of shared memory declaration and leave out size of array
            - note that they are now 1D rather than 2D as before, so need to used linearized indices
            
            ```c
            extern __shared__ Mds[];
            extern __shared__ Nds[];
            ```
            
        - at runtime, query device configurations and pass that as a 3rd configuration parameter (denoting the size of tiles) to kernel launch
            
            ```c
            size_t size = calculate_appropriate_SM_usage(dev_prop.sharedMemPerBlock, ...);
            
            matrixMulKernel<<<dimGrid, dimBlock, size>>>(M_d, N_d, P_d, Width);
            ```
            

# Summary

- execution speed of a program in modern processors can be severely limited by the speed of the memory
    - to achieve good utilization of the execution throughput of CUDA devices, a high compute-to-global-memory-access ratio in the kernel code should be obtained
    - if the ratio obtained is low, the kernel is memory-bound; i.e., its execution speed is limited by the rate at which its operands are accessed from memory
    - CUDA defines registers, shared memory, and constant memory. These memories are much smaller than the global memory but can be accessed at much higher rates. Using these memories effectively requires a redesign of the algorithm. We use matrix multiplication to illustrate tiling, a widely used technique to enhance locality of data access and effectively use shared memory. In parallel programming, tiling forces multiple threads to jointly focus on a subset of the input data at each phase
    of execution so that the subset data can be placed into these special memory types,
    consequently increasing the access speed. We demonstrate that with 16 × 16 tiling,
    global memory accesses are no longer the major limiting factor for matrix multiplication performance.
    However, CUDA programmers need to be aware of the limited sizes of these
    types of memory. Their capacities are implementation-dependent. Once their capacities are exceeded, they limit the number of threads that can simultaneously execute
    in each SM. The ability to reason about hardware limitations when developing an
    application is a key aspect of computational thinking.
    Although we introduced tiled algorithms in the context of CUDA programming,
    the technique is an effective strategy for achieving high-performance in virtually all
    types of parallel computing systems. The reason is that an application must exhibit
    locality in data access in order to effectively use high-speed memories in these systems. In a multicore CPU system, data locality allows an application to effectively
    use on-chip data caches to reduce memory access latency and achieve high-performance. Therefore, the reader will find the tiled algorithm useful when he/she develops a parallel application for other types of parallel computing systems using other
    programming models.
    Our goal for this chapter is to introduce the concept of locality, tiling, and different CUDA memory types. We introduced a tiled matrix multiplication kernel by
    using shared memory. The use of registers and constant memory in tiling has yet to
    be discussed. The use of these memory types in tiled algorithms will be explained
    when parallel algorithm patterns are discussed.

# OTHER

# Background

- now going to learn the compute and memory architectures of the modern GPU, along with performance optimization techniques reasoned from the deeper understanding of these architectures
- high-level simplified view:
    - flexible resource assignment
    - scheduling of blocks
    - occupancy
- more advanced view:
    - thread scheduling
    - latency tolerance
    - control divergence
    - synchronization

# Architecture of a modern GPU

![Screenshot 2024-09-20 at 3.47.42 PM.png](files/Lecture%204%20CUDA%20Memory%20Model/Screenshot_2024-09-20_at_3.47.42_PM.png)

- organized into an array of highly-threaded **streaming multiprocessors (SMs)**
- each SM has:
    - several processing units (hence *multi*processor), aka CUDA cores, aka **cores** (the small tiles) that share control logic and memory resources
        - eg. Ampere A100 GPU has 108 SMs with 64 cores each → total of 6912 cores in the entire GPU
    - an **on-chip memory structure** (the smaller Memory rectangle)
- GPUs also come with gigabytes of **off-chip device memory** (the larger Global Memory rectangle)
    - older GPUs used graphics double data rate synchronous DRAM
    - more recent GPUs (starting with NVIDIA’s Pascal architecture) may use HBM (high bandwidth memory) or HBM2 (DRAM modules tightly integrated with GPU in the same package)
- in this book, will refer to all these types of memory as DRAM

# Warps and SIMD hardware

- RECALL (from previous section): blocks can execute in any order relative to each other, which allows for transparent scalability across different devices → i.e. **discussed the execution timing between blocks**
- now we **discuss the execution timing of threads WITHIN each block**
- one should assume that threads within a block can execute in any order relative to each other
    - if they are running an algorithm with phases, barrier synchronizations can then be used to ensure that all threads have completed a phase before any thread starts the next phase
        - correctness of executing a kernel should not depend on any assumption that certain threads will execute in synchrony without the use of barrier synchronizations
- thread scheduling in CUDA GPUs is a hardware implementation concept → think about it in the context of specific hardware implementations (i.e. different hardware might have different thread scheduling)

## Warps

- **warp**: a unit of 32 threads (warp size is implementation specific but in most implementations it is 32 threads)
    - once a block has been assigned to an SM, it is implicitly divided into warps → a warp is the unit of thread scheduling in SMs
        
        ![There are 3 blocks all assigned to 1 SM. Each of these 3 blocks is divided into warps of 32 consecutive threads for scheduling purposes. To calculate # warps in an SM given an example block size of 256 threads: we know that each block has 256/32 = 8 warps. Then 3 blocks in the SM, we have 8 * 3 = 24 warps in the SM.](files/Lecture%204%20CUDA%20Memory%20Model/Screenshot_2024-09-21_at_10.55.58_AM.png)
        
        There are 3 blocks all assigned to 1 SM. Each of these 3 blocks is divided into warps of 32 consecutive threads for scheduling purposes. To calculate # warps in an SM given an example block size of 256 threads: we know that each block has 256/32 = 8 warps. Then 3 blocks in the SM, we have 8 * 3 = 24 warps in the SM.
        
    
    ### Block partitioning into warps
    
    - partitioned based on thread indices
    - block linearized in row-major layout with increasing `threadIdx.**z**` then increasing `threadIdx.**y**` then increasing `threadIdx.**x`** when partitioned into warps
        - eg. block organized into a 1D array (i.e. only `threadIdx.x` is used)
            - warp 0 consists of threads 0 to 31
            - warp 1 consists of threads 32 to 63
            - warp n consists of threads 32*n to 32 * (n+1) - 1
        - eg. 2D block
            
            ![Screenshot 2024-09-21 at 11.09.48 AM.png](files/Lecture%204%20CUDA%20Memory%20Model/Screenshot_2024-09-21_at_11.09.48_AM.png)
            
            - form is $T_{y,x}$
            - threadIdx.y = 0 comes before threadIdx.y = 1, etc.
                - then within threadIdx.y sections, threadIdx.x = 0 comes before threadIdx.x = 1, etc
    - for block sizes that are not a multiple of warp size 32, the last warp will be padded with inactive threads to fill up the 32 thread positions
        - eg. if a block has 48 threads, then threads 0 to 31 = 1 warp, and 32 to 47 combined with 63-47 = 16 inactive threads make up the 2nd warp

## SIMD Hardware

- an SM is designed to execute all threads in a warp following the single-instruction, multiple-data (SIMD) model → this means that at any instant in time, 1 instruction is fetched and executed for all threads in the warp

![In this example, every 8 cores (green tiles) form a processing block and share an instruction fetch/dispatch unit](files/Lecture%204%20CUDA%20Memory%20Model/Screenshot_2024-09-21_at_11.16.22_AM.png)

In this example, every 8 cores (green tiles) form a processing block and share an instruction fetch/dispatch unit

- hardware varies, eg. Ampere A100 SM has 64 cores, organized into 4 processing blocks with 16 cores each
- **threads in the same warp are assigned to the same processing block, which fetches the instruction for the warp and executes it for all threads in the warp at the same time (each thread applies the same instruction to different portions of the data)**
    - so SIMD hardware effectively restricts all threads in a warp to execute the same instruction at any point in time → so execution behavior is often referred to as single instruction, multiple thread
- advantage of SIMD is that the cost of the control hardware (eg. the instruction fetch/dispatch unit) is **SHARED** among many execution units → this allows for a smaller percentage of the hardware board to be dedicated to control, and a larger percentage to be dedicated to increasing arithmetic throughput (eg. more execution units)

## Aside: von Neumann Model

![Screenshot 2024-09-21 at 11.47.12 AM.png](files/Lecture%204%20CUDA%20Memory%20Model/Screenshot_2024-09-21_at_11.47.12_AM.png)

- blueprint model for all modern computers based on the design of the EDVAC computer
- **I/O module**: allows both programs and data to be provided to and generated from the system
- to execute a program, the computer first inputs the program and data into the **Memory module**
- a program consists of a collection of instructions
    - the **Control Unit** maintains a **program counter (PC)** which contains the memory address of the next instruction to be executed
    - in each instruction cycle, the Control Unit uses the PC to fetch and instruction into the **Instruction Register (IR)**
    - instruction bits are examined to determine the action to be taken by all components of the computer → this is why the model is also called the “stored program” model because a user can change the behavior of a computer by storing a different program into its memory

### Motivation for executing threads as warps

- consider this modified von Neumann model adapted to reflect a GPU design
    
    ![Screenshot 2024-09-21 at 12.12.33 PM.png](files/Lecture%204%20CUDA%20Memory%20Model/Screenshot_2024-09-21_at_12.12.33_PM.png)
    
- processor (corresponds to a processing block in figure 4.8) has only 1 Control Unit that fetches and dispatches instructions → these control signals then go to multiple processing units that each correspond to a core (the green tiles in figure 4.8) in the SM, each of which executes one of the threads in a warp
- since all processing units are controlled by the same instruction in the Instruction Register (IR) of the Control Unit, any execution differences are due to the different data operand values in the register files → this is called Single-Instruction-Multiple-Data (SIMD) in processor design
    - eg, although all processing units (cores) are controlled by an instruction (eg. add r1, r2, r3), the contents of r2 and r3 are different in different processing units
- Control units in modern processors are quite complex, including sophisticated logic for fetching instructions and access ports to the instruction cache.
    - so having multiple processing units to share a control unit can result in significant reduction in hardware manufacturing cost and power consumption.

# Control divergence

- SIMD execution works well when all threads within a warp follow the same execution path (i.e. control flow) when working on their data
    - however, there may be **control divergence:** when threads *within a warp* take different control flow/execution paths (eg. during an `if-else` statement)
- when there is control divergence, the SIMD hardware will take *multiple* passes through the paths (one pass for each path)
    - eg. in an `if-else` construct, if some threads in a warp follow the `if`-path, while others follow the `else`-path, the hardware will take 2 passes (one through `if`, one through `else`)
        - during each pass, the threads that follow the other path are not allowed to take effect
        
        ![Screenshot 2024-09-21 at 5.09.17 PM.png](files/Lecture%204%20CUDA%20Memory%20Model/Screenshot_2024-09-21_at_5.09.17_PM.png)
        
    - eg. in a `for-loop` construct, if some threads don’t meet the same iteration conditions and thus each thread executes a different number of loop iterations
        
        ![Screenshot 2024-09-21 at 5.25.11 PM.png](files/Lecture%204%20CUDA%20Memory%20Model/Screenshot_2024-09-21_at_5.25.11_PM.png)
        
    - this multi-pass approach to divergent warp execution extends the SIMD hardware’s ability to implement the full semantics of CUDA threads → while the hardware executes the same instruction for all threads in a warp, it selectively lets threads take effect in only the pass that corresponds to the path that they take → this allows every thread to appear to take its own control flow path
        - advantage: this preserves the independence of threads while taking advantage of the reduced cost of SIMD hardware
        - cost: extra passes and execution resources consumed by inactive threads in each pass
    - in Pascal and previous architectures, passes are executed sequentially; in Volta and onwards architectures, passes may be executed concurrently (feature referred to as independent thread scheduling)
- can determine whether a control construct can result in thread divergence by inspecting its decision condition
    - if the decision condition is based on `threadIdx` values, the control statement can **potentially** cause thread divergence (but may not…you need to look at the actual code logic)
- a prevalent reason for using a control construct with thread control divergence is handling boundary conditions when mapping threads to data
    - this is usually because the total number of threads is a multiple of the thread block size, but the size of the data can be an arbitrary number → so some threads may not be used and as such the warp that contains those thread indices will have control divergence
- **important implication of control divergence:** one can’t assume that all threads in warp will have the same execution timing (some may be inactive on a first pass and would not have completed the work yet)
    - so if all threads in a warp must complete a phase of their execution before any of them can move on, must use a barrier synchronization mechanism such as `__syncwarp()` to ensure correctness

## Performance impact of control divergence

- performance impact of control divergence decreases as the size of the vectors being processed increases
- consider a vector length of 100:
    - ceil(100/32) = 4 warps, last one will have threads 96 to 99 and a pad of 32 - 4 = 28 inactive threads, so will have control divergence
        - 1 of 4 warps will have control divergence → 25% of execution time affected → can have significant impact on performance
- consider a vector length of 1000:
    - ceil(100/32) = 32 warps, last one will have threads 992 to 999 and a pad of 32 - (999 - 992 + 1) = 24 inactive threads, so will have control divergence
        - 1 of 32 warps will have control divergence → only about 3% of execution time affected → less impact on performance
- for 2D data, the performance impact of control divergence decreases as the number of pixels in the horizontal dimension increases (this is because there would be more warps? draw it out)
    
    ![the dimension of this are irrelevant, just look at the regions](files/Lecture%204%20CUDA%20Memory%20Model/Screenshot_2024-09-21_at_5.55.09_PM.png)
    
    the dimension of this are irrelevant, just look at the regions
    
    - eg. if we process a 200 x 150 picture with 16 x 16 blocks, there will be a total of ceil(200/16) x ceil(150/16) = 13 x 10 = 130 thread blocks, which means there would be 130 * 8 = 1040 warps total
        - number of warps in regions 1: 864 (12 full blocks in horizontal dim x 9 full blocks in vertical dim x 8 warps per block)
        - number of warps in region 2: 72 (9 blocks in vertical dim hanging off edge x 8 warps per block)
        - number of warps in region 3: 96 (12 blocks in horizontal dim hanging off edge x 8 warps per block)
        - number of warps in region 4: 8 (1 block x 8 warps per block)
        - only 80 of these warps will have control divergence → so the performance impact of control
        divergence will be less than 8%

# Warp scheduling and latency tolerance

- when threads are assigned to SMs, there are usually more threads assigned to an SM than there are cores (processors/execution units) in the SM → i.e. each SM has only enough execution units to execute a subset of all the threads that are assigned to it at any point in time
    - in earlier GPU designs, each SM can execute only one instruction for a single warp at any given instant
    - in more recent GPU designs, each SM can execute instructions for a small number of warps at any given point in time
    - but in either case, the hardware can execute instructions only for a subset of all warps in the SM → so why do we need to have so many warps assigned to an SM if it can execute only a subset of them at any instant?
        - the answer: this is how GPUs tolerate long-latency operations such as global memory accesses
- when an instruction to be executed by a warp needs to wait for the result of a previously initiated long-latency operation, the warp is not selected for execution → instead, another warp that is no longer waiting for results of a previous instruction will be selected for execution
    - if more than 1 warp is ready for execution, a priority mechanism is used to choose
    - **latency tolerance:** this mechanism of filling the long-latency time of operations from some threads, with work from other threads → TLDR we need an excess of threads so that some can fill in the latent time
    - for latency tolerance to be effective, we want an SM to have many more threads assigned to it than can be simultaneously supported with its execution resources → this maximized the chance of finding a warp that is ready to execute at any point in time
        - eg. in an Ampere A100 GPU, an SM has 64 cores but can have up to 2048 threads assigned to it at the same time. Thus the SM can have up to 2048/64=32 times more threads assigned to it than its cores can support at any given clock cycle → this oversubscription of threads to SMs is essential for latency
        tolerance. It increases the chances of finding another warp to execute when a currently executing warp encounters a long-latency operation.
- warp scheduling is also used for tolerating other types of operation latencies (eg. pipelined floating-point arithmetic and branch instructions)
    - with enough warps around, the hardware will likely find a warp to execute at any point in time, which makes full use of the execution hardware while the instructions of some warps wait for the results of long-latency operations
    - **zero-overhead thread scheduling:** when the warp selection doesn’t introduce any idle or wasted time into the execution timeline
    - this ability to tolerate long operation latencies is the main reason why GPUs do not dedicate
    nearly as much chip area to cache memories and branch prediction mechanisms as CPUs do → as a result, GPUs can dedicate more chip area to floating-point execution and memory access channel resources

## Aside: Threads, Context-switching, and Zero-overhead Scheduling

- getting a deeper understanding of how threads are implemented
- a thread in modern computers is (1) a program and (2) the state of the executing the program on a von Neumann Processor → recall that it consists of the program code, the instruction in the code that is being executed, and the value of its variables and data structures
- in a computer based on the von Neumann model:
    - the code of the program is stored in memory
    - the PC (program counter) keeps track of the *address* of the next instruction in the program being executed
    - the IR (instruction register) holds the actual instruction that is being executed
    - the register and memory hold the values of the variables and data structures
- modern processors are designed to allow **context-switching:** concept where multiple threads can time-share a processor by taking turns to make progress.
    - able to do this (i.e. suspend the execution of a thread and correctly resume the execution of the thread later) by carefully saving and restoring the PC value and the contents of registers and memory
        - however, saving and restoring register contents during context-switching in these processors can incur significant overhead in terms of added execution time.
- **Zero-overhead scheduling:** refers to the GPU’s ability to put a warp that needs to wait for a long-latency instruction result to sleep and activate a warp that is ready to go without introducing any extra idle cycles in the processing units
    - Traditional CPUs incur such idle cycles because switching the execution from one thread to another requires saving the execution state (such as register contents of the out-going thread) to memory and
    loading the execution state of the incoming thread from memory.
    - GPU SMs achieves zero-overhead scheduling by holding all the execution states for the assigned warps in the hardware registers so there is no need to save and restore states when switching from one warp to another.

# Resource partitioning and occupancy

- RECAP: in order for long-latency operations to be tolerated, we want to assign as many warps to an SM than can be simultaneously supported with its execution resources
    - but it may not always be possible to assign to the SM the maximum number of warps that the SM supports
- **occupancy:** the ratio of $\frac{\text{the number of warps ASSIGNED to an SM}}{\text{the MAXIMUM number of warps the SM supports}}$
    - things might prevent an SM from reaching maximum occupancy

## Resource partitioning

- the execution resources in an SM include:
    - registers
    - shared memory
    - thread block slots
    - thread slots
- these execution resources are dynamically partitioned across threads
    - eg. an Ampere A100 GPU can support a maximum of 32 blocks per SM, 64 warps (64 * 32 = 2048 threads) per SM, and 1024 threads per block.
        - If a grid is launched with a block size of 1024 threads (recall from quiz 0 that this is the maximum allowed), the 2048 thread slots per SM are partitioned and assigned to 2 blocks.
            - So in this case, each SM can accommodate up to 2 blocks.
        - Similarly, if a grid is launched with a block size of 512, 256, 128, or 64 threads, the 2048 thread slots are partitioned and assigned to 2048/512 = 4,  2048/256 = 8,  2048/128 = 16, or  2048/64 = 32 blocks, respectively
    - this ability to dynamically partition thread slots makes SMs versatile → they can either execute many blocks each having few threads or execute few blocks each having many threads
        - contrast: fixed partitioning where each block receives a fixed amount of resources (eg. thread slots) regardless of how many threads it has → results in wasted thread slots when one block requires fewer thread slots than the fixed partition supports, but another block requires more than the threads slots provided
- dynamic partitioning of resources can lead to subtle interactions between resource limitations (eg. between block slots and thread slots), which can cause an underutilization of resources
    - eg. in the Ampere A100 architecture, a block size can be varied from 1024 threads (2 blocks per SM) to 64 threads (1024/max 32 blocks per SM = 64)
        - in all these cases, the total number of threads assigned to the SM is 2048 → maximizes occupancy
            - consider a different case when each block has 32 threads → in this case, the 2048 thread slots would need be partitioned and assigned to 2048/32 = 64 blocks
                - now consider the Volta SM that can only support 32 block slots at once → given the above configuration, this means that only half the thread slots will be utilized (i.e. 32 blocks with 32 threads each) → occupancy is $\frac{\text{1024 assigned threads}}{\text{2048 maximum threads}} = 50$%.
                    - so to fully utilize the thread slots and achieve maximum occupancy, one needs at least 64 threads in each block
- **another situation that could negatively affect occupancy occurs when the maximum number of threads per block is not divisible by the block size**
    - eg. consider Ampere A100 again: we saw that up to 2048 threads per SM can be supported. However, if a block size of 768 is selected, the SM will be able to accommodate only floor(2048/768) = 2 full thread blocks (768 * 2 = 1536 threads), leaving 2048 - 1536 = 512 thread slots unutilized
        - in this case, neither the maximum threads per SM nor the maximum blocks per SM are reached. The occupancy in this case is (1536 assigned threads)/(2,048 maximum threads) = 75%.
- there are also other resource constraints from as registers and shared memory
    - automatic variables declared in a CUDA kernel are placed into registers → some kernels may use many automatic variables, and others may use few of them → so one should expect that some kernels require many registers per thread and some require few.
        - By dynamically partitioning registers in an SM across threads, the SM can accommodate many blocks if they require few registers per thread and fewer blocks if they require more registers per thread
        - however, need to be aware of potential impact of register resource limitations on occupancy.
            - eg. the Ampere A100 GPU allows a maximum of 65,536 registers per SM. To run at full occupancy, each SM needs enough registers for 2048 threads, which means that each thread should not use more than (65,536 registers)/(2048 threads) = 32 registers per thread
                - if a kernel uses 64 registers per thread, the maximum number of threads that can be supported with 65,536 registers is 1024 threads. In this case, the kernel cannot run with full occupancy regardless of what the block size is set to be → instead, the occupancy will be at most 50%.
                - In some cases, the compiler may perform register spilling to reduce the register requirement per thread and thus elevate the level of occupancy. However, this is typically at the cost of increased execution time for the threads to access the spilled register values from memory and may cause the total execution time of the grid to increase.

## Example

- Assume that a programmer implements a kernel that uses 31 registers per thread and configures it with 512 threads per block. In this case, the SM will have (2048 max threads)/(512 threads/block) = 4 blocks running simultaneously. These threads will use a total of (2048 threads) x (31 registers/thread) = 63,488 registers, which is less than the 65,536 register limit.
- Now assume that the programmer declares another two automatic variables in the kernel, bumping the number of registers used by each thread to 33. The number of registers required by 2048 threads is now 67,584 registers, which exceeds the register limit.
- The CUDA runtime system may deal with this situation by assigning only 3 blocks to each SM instead of 4, thus reducing the number of registers required to 50,688 registers.
    - However, this reduces the number of threads running on an SM from 2048 to 1536
    - so by using two extra automatic variables, the program saw a reduction in occupancy from 100% to 75%. This is sometimes referred to as a “**performance cliff,**” in which a slight increase in resource usage can result in significant reduction in parallelism and performance achieved (Ryoo et al., 2008).

### Summary

- the constraints of all the dynamically partitioned resources interact with each other in a complex manner. Accurate determination of the number of threads running in each SM can be difficult.

# Summary

- programs are executed in warp groupings, meaning that threads in a single warp are executing the same instructions at a time. Control divergence happens within the executing warp, there CAN be different conditional paths within the same GRID without control divergence
- A GPU is organized into SM, which consist of multiple processing blocks of cores that share control logic and memory resources.
- When a grid is launched, its blocks are assigned to SMs in an arbitrary order, resulting in **transparent scalability** of CUDA applications.
    - The transparent scalability comes with a limitation: Threads in different blocks cannot synchronize with each other.
- Threads are assigned to SMs for execution on a block-by-block basis. Once a block has been assigned to an SM, it is further partitioned into warps. Threads in a warp are executed following the SIMD model. If threads in the same warp diverge by taking different execution paths, the processing block executes these paths in passes in which each thread is active only in the pass corresponding to the path that it takes.
- An SM may have many more threads assigned to it than it can execute simultaneously. At any time, the SM executes instructions of only a small subset of its resident warps. This allows the other warps to wait for long-latency operations without slowing down the overall execution throughput of the massive number of processing units.
    - The ratio of the number of threads assigned to the SM to the maximum number of threads it can support is referred to as **occupancy**
        - the higher the occupancy of an SM, the better it can hide long-latency operations.
- Each CUDA device imposes a potentially different limitation on the amount of resources available in each SM. For example, each CUDA device has a limit on the number of blocks, the number of threads, the number of registers, and the amount of other resources that each of its SMs can accommodate. For each kernel, one or more of these resource limitations can become the limiting factor for occupancy. CUDA C provides programmers with the ability to query the resources available in a GPU at runtime.

# Resources

- Kirk & Hwu: Chapter 4 Memory and data locality