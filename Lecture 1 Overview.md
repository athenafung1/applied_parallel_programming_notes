# Lecture 1: Introduction to the Course

Reviewed: Yes
Textbook Chapter: 1

# Background

- in 1980s/1990s: microprocessor based on a **single** **CPU** (eg. x86 processors from Intel and AMD) drove rapid performance increases and cost reductions in computer applications due to fast increasing clock frequency and hardware resources
    - this drive slowed down since 2003 due to energy consumption and heat dissipation issues that limit the increase of clock frequency and the activities that can be performed in each clock period within a **single** **CPU** while maintaining the appearance of executing instructions in sequential steps
- virtually all microprocessors then switched to a model with **multiple CPUs** (referred to as processor cores) in each chip
    - to benefit from the multiple processor cores, users need to have the ability to execute multiple instruction sequences (whether from the same application or different applications) simultaneously → i.e. the need to parallelize
- **parallel programs:** multiple threads of execution cooperate to complete the work faster

# 1.1 Heterogeneous parallel computing

- since 2003, there have been 2 main trajectories for the semiconductor industry:
    - **multicore**: seeks to maintain the execution speed of sequential programs while moving into multiple cores
        - **ASIDE:** multicore vs. multiprocessor: The primary difference between multicore and multiprocessor is that a **multicore operates a single CPU with multiple individual processing units**, while a **multiprocessor has multiple CPUs**. In a simplified way, if you want a computer to run a single program faster, that is a job for a multicore processor. If you need multiple programs run simultaneously, then a multiprocessor CPU is the way to go.
    - **many-thread**: focuses on the execution throughput of parallel applications
        - eg. NVIDIA Tesla A100 GPUs support tens of thousands of threads executing in simple, in-order pipelines
        - many-thread processors, especially GPUs have led the race of floating-point performance
    - there is a large gap in peak performance between multicore CPUs and many-thread GPUs that has driven application developers to move the computationally intensive parts of their software to GPUs for execution
        - this difference is due to the fundamental design philosophies between the 2 types of processors:
            
            ![GPU design has a larger number of smaller ALUs (green) and memory channels (gray)](files/Lecture%201%20Overview/Screenshot_2024-09-05_at_11.40.02_AM.png)
            
            GPU design has a larger number of smaller ALUs (green) and memory channels (gray)
            
            - **CPUs** are optimized for sequential code performance by having the CPU hardware reduces the execution latency of each individual thread by reducing the latency of operations:
                - ALUs and operand data delivery logic are designed to minimize the effective latency of arithmetic operations
                - Large last-level on-chip caches are designed to capture frequently accessed data and convert some of the long-latency memory accesses into short-latency cache accesses
                - Sophisticated branch prediction logic and execution control logic are used to mitigate the latency of conditional branch instructions
                - but all this at the cost of **increased use of chip area and power per unit that could otherwise be used to provide more ALUs and memory access channels**
            - **GPUs**
                - shaped by the video-game industry where massive number of floating-point calculations and memory accesses per video frame is needed to advance games → GPU vendors look for ways to maximize chip area and power budget dedicated to FP calculations and memory access throughput
                - a GPU needs to be capable of moving extremely large amounts of data into and out of graphics frame buffers in its DRAM (dynamic random-access memory)
        - **important observation:** reducing latency (CPU design) is much more expensive than increasing throughput (GPU design) in terms of power and chip area
            - eg. one can double the arithmetic throughput by doubling the number of arithmetic units at the cost of doubling chip area and power consumption BUT reducing arithmetic latency by half may require doubling the current at the cost of more than doubling the chip area used and quadrupling the power consumption
            - **this means that the prevailing incentive with GPUs is to optimize for the execution throughput of massive numbers of threads rather than reducing the latency of individual threads** → saves chip area and power by allowing pipelined memory channels and arithmetic operations to have long latency
                - reduction in area and power of the memory access hardware and arithmetic units allows the GPU designers to have more of them on a chip and thus increase the total execution throughput

## More on the throughput-oriented design (GPUs)

- **strives to maximize the total execution throughput of a large number of threads while allowing individual threads to take a potentially much longer time to execute**
    - application software for GPUs is expected to be written with a large number of parallel threads because the hardware takes advantage of the large number of threads to find work to do when some of them are waiting for long-latency memory accesses or arithmetic operations
    - small cache memories help control bandwidth requirements so that multiple threads that access the same memory data do not all need to go to the DRAM
- GPUs are designed as parallel, throughput-oriented computing engines and as such will not perform well on some tasks that CPUs perform well on
    - eg. tasks that have very few threads → CPUs w/lower operation latencies will achieve much higher performance
    - so, many applications will utilize both CPUs (for the sequential parts) and GPUs (for the numerically intensive parts) → **NVIDIA’s Compute Unified Device Architecture (CUDA) programming model was designed to support the joint CPU-GPU execution of an application**

## Other important things to note

- speed is not the only decision factor when developers choose the processors for running their applications:
    - **installed base**: the ***number of units that are currently in use by customers** →* provides a measurement of a company's existing customer base.
        - the processors of choice must have a very large installed base → simply because applications developed to be run on a processor with a large market presence will have a large customer base
        - because of the popularity of many-thread GPUs in the PC market, they have a large market presence and as such they are economically attractive targets for application developers (TLDR there is an incentive to use CUDA-enabled GPUs when developing applications)
    - **practical form factors**: a hardware design aspect that defines and prescribes the size, shape, and other physical specifications of components
        - should have reasonably attainable form factors (eg. number of nodes in a cluster)
    - **easy accessibility**
        - until 2006, graphic chips were difficult to use because programmers had to use the equivalent of graphics API functions to access the processing units → a computation must be expressed as a function that paints a pixel in some way in order to execute on these early GPUs
            - not very practical for generic applications to have to reduce to pixel painting
- CUDA represented software AND hardware changes
    - software layers were redone so that familiar C/C++ programming tools could be used
    - NVIDIA actually devoted silicon area to facilitate the ease of parallel programming with a **new general-purpose parallel programming interface on the silicon chip that serves the requests of CUDA programs**
        - so general-purpose GPU programs no longer had to go through graphics interfaces at all, and this expanded the types of applications that one can develop for GPUs

# 1.2 Why more speed or parallelism?

- supercomputing applications, simulations (eg. on the molecular level for observations that can’t be done with tradition optics/electronic instrumentation), high-resolution displays, better user interfaces, consumer electronic gaming
- **new applications:** deep learning based on artificial neural networks

# 1.3 Speeding up real applications with parallelization

- **speedup** (for an application by computing system A over computing system B): $\frac{\text{time used to execute the application in system B}}{\text{time used to execute the application in system A}}$
    - eg. if an application takes 10 seconds to execute in system A but takes 200 seconds to execute in system B, the speedup for the execution by system A over system B would be $\frac{200}{10}=20$x speedup.
- factors affecting the achievable levels of speedup:
    - **the portion of code that can be parallelized**
        - **Amdahl’s Law**: the speedup that is achievable by a parallel computing system *over* a serial computing system can be severely limited by the portion of the application that can be parallelized
            - important that an application has the VAST MAJORITY of its execution in the parallel portion in order for a massively parallel processor to effectively speed up execution
    - **how fast the data can be accessed from and written to the memory**
        - in practice, straightforward parallelization of applications often saturates the memory (DRAM) bandwidth, resulting in only a 10x speedup
        - trick is to figure out how to get around memory bandwidth limitations → involves doing transformations to utilize specialized GPU on-chip memories in order to reduce the number of accesses to the DRAM, but also need to get around the limitation of limited on-chip memory capacity
- the level of speedup that is achieved over single-core CPU execution can also reflect the suitability of the CPU to the application → if CPUs perform well for some application, it will be harder to speed up performance by using a GPU
    - most applications will have portions that can be better executed by the CPU, so it is important to write code such that we have a **heterogenous parallel computing capability of a combined CPU/GPU system where the GPUs *complement* the CPU execution**

# 1.4 Challenges in parallel programming

1. challenging to design parallel algorithms with the same level of algorithmic (computational) complexity as that of sequential algorithms
    - **many parallel algorithms perform the same amount of work as their sequential counterparts, but some do MORE work** (even as to end up slower for large input datasets)
    - eg. many real-world problems are most naturally described with mathematical recurrences → parallelizing these problems requires nonintuitive reframing of the problem and potentially redundant work
2. execution speed of many applications is limited by memory access latency and/or throughput → these types of applications are referred to as “memory bound” (in contrast, “compute bound” applications are limited by the number of instructions performed per byte of data)
    - **achieving high-performance parallel execution in memory-bound applications requires methods for improving memory access speed**
3. execution speed of parallel programs is often **more sensitive to the input data characteristics** (compared to sequential counterparts)
    - variations in data sizes and data distributions may cause uneven amounts of work to be assigned to the parallel threads → significantly reduces the effectiveness of the parallel execution
4. **synchronization operations impose overhead on the application because threads may be waiting on other threads instead of doing useful work**
    - some applications require threads to collaborate with each other → requires using synchronization operations such as barriers or atomic operations
    - in contrast, “embarrassingly parallel” applications can be parallelized without much collaboration across different threads

# 1.5 Related parallel programming interfaces

- OpenMP for shared memory multiprocessor systems
    - an OpenMP implementation consists of a compiler and a runtime
        - programmer specifies directives (commands) and pragmas (hints) about a loop to the OpenMP compiler. With these directives and pragmas, OpenMP compilers generate parallel code. The runtime system supports the execution of the parallel code by managing parallel threads and resources
    - OpenMP was originally designed for CPU execution and has been extended to support GPU execution
    - **major advantage**: it provides compiler automation and runtime support for abstracting away many parallel programming details from programmers.
        - Such automation and abstraction can help to make the application code more
        portable across systems produced by different vendors as well as different generations of systems from the same vendor → this is **performance
        portability**
    - **disadvantage**: effective programming in OpenMP still requires the programmer to understand all the detailed parallel programming concepts that are involved
- Message Passing Interface (MPI) for scalable cluster computing
    - MPI is a programming interface in which computing nodes in a cluster do not share memory → all data sharing and interaction must be done through explicit message passing
    - widely used in HPC → applications written in MPI have run successfully on cluster computing systems with more than 100,000 nodes
        - today, many HPC clusters employ heterogeneous CPU/GPU nodes
    - due to the lack of shared memory across computing nodes, the amount of effort that is needed to port an application into MPI can be quite high → programmer needs to perform domain decomposition to partition the input and output data across individual nodes. On the basis of the domain decomposition, the programmer also needs to call message sending and receiving functions to manage the data exchange between nodes
        - CUDA, by contrast, provides shared memory for parallel execution in the GPU to address this difficulty. While CUDA is an effective interface with each node, most application developers need to use MPI to program at the cluster level. Furthermore, there has been increasing support for multi-GPU programming in CUDA via APIs such as the NVIDIA Collective Communications Library (NCCL) → therefore important that a parallel programmer in HPC understands how to do joint MPI/CUDA programming in modern computing clusters employing multi-GPU nodes
- OpenC(ompute)L(anguage): programming model that defines language extensions and runtime APIs allowing programmers to manage parallelism and data delivery in massively parallel processors

# 1.6 Overarching Goals

- need to have a good conceptual understanding of the parallel hardware architectures to be able to reason about the performance behavior of your code
- know the fundamentals of the GPU architecture
- CUDA programming model encourages the use of simple forms of barrier synchronization, memory consistency, and atomicity for managing parallelism. In addition, it provides an array of powerful tools that allow one to debug not only the functional aspects, but also the performance bottlenecks. We will show that by focusing on data parallelism, one can achieve both high performance and high reliability in one’s applications.
- scalability across future hardware generations → key to such scalability is to regularize and localize memory data accesses to minimize consumption of critical resources and conflicts in updating data structures

## Key Thought Processes for CUDA programming

1. identifying the part of application programs to be parallelized
2. isolating the data to be used by the parallelized code, using an API function to allocate memory on the parallel computing device
3. using an API function to transfer data to the parallel computing device
4. developing the parallel part into a kernel function that will be executed by parallel threads
5. launching a kernel function for execution by parallel threads
6. eventually transferring the data back to the host processor with an API function call

# Resources

- Kirk & Hwu: Chapter 1
- NVIDIA CUDA Docs: [https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#introduction](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#introduction)
- Lecture Slides: [https://canvas.illinois.edu/courses/49985/files/folder/lectures?preview=13426897](https://canvas.illinois.edu/courses/49985/files/folder/lectures?preview=13426897)
- Lecture Video: [https://mediaspace.illinois.edu/media/t/1_03az0ub4/351036722](https://mediaspace.illinois.edu/media/t/1_03az0ub4/351036722)