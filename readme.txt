Parallelizing multiple precision Taylor series method for integrating the Lorenz system

To run our hybrid programs one needs MPIGMP library of Tomonori Kouya. This library can be freely downloaded from:   
http://na-inet.jp/na/bnc/

-----------------------------------------------------------------------------------

hybrid1.c program computes x[i+1],y[i+1],z[i+1] and  x[0],y[0], z[0] independently in parallel but do not overlap MPI_ALLREDUCE with the operations for  x[i+1],y[i+1],z[i+1] that can be taken in advance. The parallel computation of x[i+1],y[i+1],z[i+1] and x[0],y[0],z[0]
is a small and limited parallelism but it is important, because it reduces the serial part of the code and improves the speedup from the Amdahl's low. 

------------------------------------------------------------------------------------

hybrid2.c program is an improvement of hybrid1.c  MPI_ALLREDUCE is overlapped with some computations for x[i+1], y[i+1], z[i+1] that can be taken in advance before the computation of the sums s1 and s2 is finished. This is nothing else but additional reduction of the serial part of the code and improvement of the speedup from the Amdahl's low.

----------------------------------------------------------------------------------

hybrid3.c program is an improvement of hybrid2.c 
Using SPMD programming pattern we make half of the threads to compute the sum s1 and the other half to compute the sum s2. We have a little performance benefit, because for the small values of the index i unused threads will be less and also the difference from the perfect load balance between threads will be less. However this approach is not general, because it strongly depends on the number of sums for reduction (two in the particular case of the Lorenz system) and the number of available threads.

It is important to note that for our problem the pure OpenMP parallelization has its own importance. First, the programming with OpenMP is easier because it avoids the usage of libraries like MPIGMP. Second, since the algorithm does not allow domain decomposition, the memory needed for one computational node is multiplied by the number of the MPI processes per that node, while OpenMP needs only one copy of the computational domain and thus some memory is saved. For completeness, we also give three pure OpenMP programs.

---------------------------------------------------------------------------------------

OpenMP1.c program computes x[i+1],y[i+1],z[i+1] and  x[0],y[0], z[0] independently in parallel. We made manually a standard tree based parallel reduction and the number of stages is only logarithm of the number of threads. This program uses #pragma omp for for work
sharing between threads.

----------------------------------------------------------------------------------------

OpenMP2.c program is the same as OpenMP1.c but uses SPMD pattern for work sharing between threads

---------------------------------------------------------------------------------------

OpenMP3.c uses SPMD programming pattern and make half of the threads to compute the sum s1 and the other half to compute the sum s2. We have a little performance benefit, because for the small values of the index i unused threads will be less and also the difference from the perfect load balance between threads will be less.
