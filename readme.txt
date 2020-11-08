Parallelizing multiple precision Taylor series method for integrating the Lorenz system

------------------------------------------------------------------------------
To run our programs one needs MPIGMP library of Tomonori Kouya. This library can be freely downloaded from: 
http://na-inet.jp/na/bnc/
------------------------------------------------------------------------------
hybrid1.c program computes x[i+1],y[i+1],z[i+1] and  x[0],y[0], z[0] independently in parallel but do not 
overlap MPI_ALLREDUCE with the operations for x[i+1],y[i+1],z[i+1] that can be taken in advance.
-----------------------------------------------------------------------------
hybrid2.c program is improvement of hybrid1.c  MPI_ALLREDUCE is overlapped with some computations for 
x[i+1], y[i+1], z[i+1] that can be taken in advance before the computation of the sums s1 and s2 is finished. 
This is nothing else but small improvement of Amdahl's low.
------------------------------------------------------------------------------
hybrid3.c program is improvement of of hybrid2.c 
Using SPMD progarmming pattern we make half of the threads to compute the sum s1 and the other half to compute
the sum s2. We have a little performance benefit, because for the small values of the index i unused threads 
will be less and also the difference from the perfect load balance between threads will be less. 
This is again nothing else but small improvement of Amdahl's low.
