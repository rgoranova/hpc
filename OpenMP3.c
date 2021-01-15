#include <stdio.h>
#include <gmp.h>
#include <omp.h>
#define PREC 11648
#define N 2800
#define Numt 32
#define pad 8


int main()
{
    FILE *outfile;
    outfile = fopen("output_OpenMP3.txt", "w");
    int i,j,l,tid,n,shift,sh,ifinal,istart,nt;

    mpf_set_default_prec(PREC);
    mpf_t R, Sigma, b, tau,zero;
    mpf_t h1,h2,h3;
    mpf_t u1,u2,time,T;
    mpf_init(h1);
    mpf_init(h2);
    mpf_init(h3);
    mpf_init(T);


    mpf_init_set_str(b,"0.0",10);
    mpf_init_set_str(zero,"0.0",10);
    mpf_init_set_str(time,"0.0",10);
    mpf_init_set_str(tau,"0.01",10);
    mpf_init_set_str(R,"28.0",10);
    mpf_init_set_str(Sigma,"10.0",10);
    mpf_init_set_str(u1,"8.0",10);
    mpf_init_set_str(u2,"3.0",10);
    mpf_div(u1,u1,u2);
    mpf_set(b,u1);
    mpf_clear(u1);
    mpf_clear(u2);

    mpf_t x[N+1],y[N+1],z[N+1];

    for (i = 0; i<N+1; i++)
    {
       mpf_init(x[i]);
       mpf_init(y[i]);
       mpf_init(z[i]);

    }

    mpf_t sum[pad*Numt],tempv[pad*Numt];
    #pragma omp parallel for schedule(static)
    for (i = 0; i<pad*Numt; i++)
    {
        mpf_init_set(tempv[i],zero);
    }

    #pragma omp parallel for schedule(static)
    for (i = 0; i<pad*Numt; i++)
    {
        mpf_init_set(sum[i],zero);
    }

    mpf_set_str(T,"1.001",10);
    mpf_set_str(x[0],"-15.8",10);
    mpf_set_str(y[0],"-17.48",10);
    mpf_set_str(z[0],"35.64",10);

    mpf_set_str(time,"0.0",10);
    double start = omp_get_wtime();

    l=0;
    while (mpf_cmp(time,T)<0)
    {
    l++;
    #pragma omp parallel private(i,j,tid,n,shift,nt,sh,ifinal,istart)
    {
        tid = omp_get_thread_num();
        nt =  omp_get_num_threads()/2;
        for (i = 0; i<N; i++)
        {
            if(tid<nt)
            {
            istart=(tid*(i+1))/nt;
            ifinal=((tid+1)*(i+1))/nt -1;

            for (j=istart; j<=ifinal; j++)
            {
                mpf_mul(tempv[pad*tid],x[i-j],z[j]);
                mpf_add(sum[pad*tid],sum[pad*tid],tempv[pad*tid]);
            }
            }
            else
            {
            istart=((tid-nt)*(i+1))/nt;
            ifinal=((tid-nt+1)*(i+1))/nt -1;

            for (j=istart; j<=ifinal; j++)
            {
                mpf_mul(tempv[pad*tid],x[i-j],y[j]);
                mpf_add(sum[pad*tid],sum[pad*tid],tempv[pad*tid]);

            }
            }
            # pragma omp barrier
            //! Explicit Parallel Reduction for two sums for log(p) additions
            //! The first step is in a butterfly form
            sh=nt;
            n=nt;
            shift=(n+1)/2;
            while (n>1)
            {
                  if (tid <=n-1-shift)
                  {
                        mpf_add(sum[pad*tid],sum[pad*tid],sum[pad*(tid+shift)]);
                  }
                  else if (tid>=sh && tid<=sh+n-1-shift)
                  {
                        mpf_add(sum[pad*tid],sum[pad*tid],sum[pad*(tid+shift)]);
                  }
                  n=shift;
                  shift=(n+1)/2;
                  # pragma omp barrier
            }
            /// End of explicit Parallel Reduction for two sums for log(p) additions
             #pragma omp sections
             {
             #pragma omp section
             {
             mpf_sub(h1,y[i],x[i]);
             mpf_div_ui(h1,h1,i+1);
             mpf_mul(x[i+1],h1,Sigma);
             }

             #pragma omp section
             {
             mpf_mul(h2,R,x[i]);
             mpf_sub(h2,h2,y[i]);
             mpf_sub(h2,h2,sum[0]);
             mpf_div_ui(y[i+1],h2,i+1);
             }

             #pragma omp section
             {
             mpf_mul(h3,b,z[i]);
             mpf_sub(h3,sum[pad*sh],h3);
             mpf_div_ui(z[i+1],h3,i+1);
             }
             }

             mpf_set(sum[pad*tid],zero);
        }



        //! One step forward with Horner's rule
        #pragma omp sections
        {
              #pragma omp section
              {
                     mpf_set(h1,x[N]);
                     for (j=N-1; j>=0; j--)
                     {
                         mpf_mul(h1,h1,tau);
                         mpf_add(h1,h1,x[j]);
                     }
                     mpf_set(x[0],h1);
              }
              #pragma omp section
              {
                     mpf_set(h2,y[N]);
                     for (j=N-1; j>=0; j--)
                     {
                         mpf_mul(h2,h2,tau);
                         mpf_add(h2,h2,y[j]);
                     }
                     mpf_set(y[0],h2);
              }
              #pragma omp section
              {
                     mpf_set(h3,z[N]);
                     for (j=N-1; j>=0; j--)
                     {
                         mpf_mul(h3,h3,tau);
                         mpf_add(h3,h3,z[j]);
                     }
                     mpf_set(z[0],h3);
              }

        }
    }

        mpf_add(time,time,tau);
        if(l==5)
        {
             mpf_out_str(outfile,10,4200,x[0]);
             fprintf(outfile,"\n");
             mpf_out_str(outfile,10,4200,y[0]);
             fprintf(outfile,"\n");
             mpf_out_str(outfile,10,4200,z[0]);
             fprintf(outfile,"\n");
             l=0;
        }

    }

    printf("Time = %f.\n",omp_get_wtime()-start);

    mpf_clear(h1);
    mpf_clear(h2);
    mpf_clear(h3);
    mpf_clear(T);
    mpf_clear(zero);
    mpf_clear(time);
    mpf_clear(tau);
    mpf_clear(R);
    mpf_clear(Sigma);
    mpf_clear(b);

    for (i = 0; i<N+1; i++)
    {
        mpf_clear(x[i]);
        mpf_clear(y[i]);
        mpf_clear(z[i]);
    }

    for (i = 0; i<pad*Numt; i++)
    {
        mpf_clear(tempv[i]);

    }

    for (i = 0; i<pad*Numt; i++)
    {
        mpf_clear(sum[i]);

    }

    fclose(outfile);


    return 0;
}






