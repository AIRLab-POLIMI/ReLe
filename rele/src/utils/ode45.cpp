/*
 * MEX C file for ODE integration (Dorman-Prince RK45)
 * Alessandro Falsone
 * Politecnico di MIlano - 2014
 *
 */

#include "ode45.h"

#include <stdio.h>      /* printf */
#include <math.h>       /* isinf, sqrt */
#include <string.h>     /* memcpy */

/* ######## Runge-Kutta Dormand-Prince 4(5) Coefficients ######## */
#define		A21		((double) 1/5)
#define		A31		((double) 3/40)
#define		A32		((double) 9/40)
#define		A41		((double) 44/45)
#define		A42		((double) -56/15)
#define		A43		((double) 32/9)
#define		A51		((double) 19372/6561)
#define		A52		((double) -25360/2187)
#define		A53		((double) 64448/6561)
#define		A54		((double) -212/729)
#define		A61		((double) 9017/3168)
#define		A62		((double) -355/33)
#define		A63		((double) 46732/5247)
#define		A64		((double) 49/176)
#define		A65		((double) -5103/18656)
#define		A71		((double) 35/384)
#define		A72		((double) 0)
#define		A73		((double) 500/1113)
#define		A74		((double) 125/192)
#define		A75		((double) -2187/6784)
#define		A76		((double) 11/84)

#define		B61		((double) 35/384)
#define		B62		((double) 0)
#define		B63		((double) 500/1113)
#define		B64		((double) 125/192)
#define		B65		((double) -2187/6784)
#define		B66		((double) 11/84)
#define		B71		((double) 5179/57600)
#define		B72		((double) 0)
#define		B73		((double) 7571/16695)
#define		B74		((double) 393/640)
#define		B75		((double) -92097/339200)
#define		B76		((double) 187/2100)
#define		B77		((double) 1/40)

#define		C1		((double) 0)
#define		C2		((double) 1/5)
#define		C3		((double) 3/10)
#define		C4		((double) 4/5)
#define		C5		((double) 8/9)
#define		C6		((double) 1)
#define		C7		((double) 1)

/* ######## Solver Parameters ######## */
#define		hMIN		((double) 1e-10)
#define		hMAX		((double) 1)
#define		RTOL		((double) 1e-3)
/*#define		ATOL		((double) 1e-6)*/
#define		FRAC		((double) 0.8)
#define		POW			((double) 1/5)
#define		THRESHOLD	((double) 1e-3)

void ode45_stepper(ode_function f,
                   void *fparams,
                   bool variable_step,
                   double h,
                   double ti,
                   double tf,
                   const double *x0,
                   unsigned int nos,
                   double **time,
                   double **state,
                   unsigned int *ndata)
{
    /* Allocate dynamic memory */
    double t = ti;
    double *x = (double*) calloc(nos,sizeof(double));
    memcpy(x,x0,nos*sizeof(double));

    double *X1 = (double*) calloc(nos,sizeof(double));
    double *X2 = (double*) calloc(nos,sizeof(double));
    double *X3 = (double*) calloc(nos,sizeof(double));
    double *X4 = (double*) calloc(nos,sizeof(double));
    double *X5 = (double*) calloc(nos,sizeof(double));
    double *X6 = (double*) calloc(nos,sizeof(double));
    double *X7 = (double*) calloc(nos,sizeof(double));
    double *x1 = (double*) calloc(nos,sizeof(double));
    double *x2 = (double*) calloc(nos,sizeof(double));
    double *err = (double*) calloc(nos,sizeof(double));

    double *F21 = (double*) calloc(nos,sizeof(double));
    double *F31 = (double*) calloc(nos,sizeof(double));
    double *F32 = (double*) calloc(nos,sizeof(double));
    double *F41 = (double*) calloc(nos,sizeof(double));
    double *F42 = (double*) calloc(nos,sizeof(double));
    double *F43 = (double*) calloc(nos,sizeof(double));
    double *F51 = (double*) calloc(nos,sizeof(double));
    double *F52 = (double*) calloc(nos,sizeof(double));
    double *F53 = (double*) calloc(nos,sizeof(double));
    double *F54 = (double*) calloc(nos,sizeof(double));
    double *F61 = (double*) calloc(nos,sizeof(double));
    double *F62 = (double*) calloc(nos,sizeof(double));
    double *F63 = (double*) calloc(nos,sizeof(double));
    double *F64 = (double*) calloc(nos,sizeof(double));
    double *F65 = (double*) calloc(nos,sizeof(double));
    double *F71 = (double*) calloc(nos,sizeof(double));
    double *F72 = (double*) calloc(nos,sizeof(double));
    double *F73 = (double*) calloc(nos,sizeof(double));
    double *F74 = (double*) calloc(nos,sizeof(double));
    double *F75 = (double*) calloc(nos,sizeof(double));
    double *F76 = (double*) calloc(nos,sizeof(double));

    double *F11 = (double*) calloc(nos,sizeof(double));
    double *F22 = (double*) calloc(nos,sizeof(double));
    double *F33 = (double*) calloc(nos,sizeof(double));
    double *F44 = (double*) calloc(nos,sizeof(double));
    double *F55 = (double*) calloc(nos,sizeof(double));
    double *F66 = (double*) calloc(nos,sizeof(double));
    double *F77 = (double*) calloc(nos,sizeof(double));

    /* Initialize solver */
    unsigned int i = 0;
    unsigned int j = 0;
    (*time)[0] = t;
    for(j; j<nos; j++)
    {
        (*state)[j] = x[j];
    }

//    const ode_function f = *(fun);
//    double hMAX = (tf-ti)/10;

    /* Guess the initial step size */
    if (variable_step)
    {
        if (hMAX <= tf-ti)
        {
            h = hMAX;
        }
        else
        {
            h = tf-ti;
        }
        double rh = 0;
        double *F0 = (double*) calloc(nos,sizeof(double));
        f(ti,x0,F0,fparams);
        for(j=0; j<nos; j++)
        {
            if (fabs(x0[j]) < THRESHOLD)
            {
                F0[j] = fabs(F0[j]/THRESHOLD);
            }
            else
            {
                F0[j] = fabs(F0[j]/x0[j]);
            }
            if (F0[j] > rh)
            {
                rh = F0[j];
            }
        }
        rh = rh/(0.8*pow(RTOL,POW));
        free(F0);
        if (h*rh > 1)
        {
            h = 1/rh;
        }
    }

    /* Main Loop */
    while(t < tf)
    {
        if (t+1.1*h >= tf)
        {
            h = tf-t;
        }
        if (variable_step)
        {
            *time = (double*) realloc(*time,(i+2)*sizeof(double));
            *state = (double*) realloc(*state,nos*(i+2)*sizeof(double));
        }

        /* Repeat until the step size ensures tolerance satisfaction */
        bool reject_solution = true;
        while(reject_solution)
        {
            reject_solution = false;

            /* Computing X1 */
            memcpy(X1,x,nos*sizeof(double));

            /* Computing X2 */
            f(t+C2*h,X1,F21,fparams);
            for(j=0; j<nos; j++)
            {
                X2[j] = x[j] + h*(A21*F21[j]);
            }

            /* Computing X3 */
            f(t+C3*h,X1,F31,fparams);
            f(t+C3*h,X2,F32,fparams);
            for(j=0; j<nos; j++)
            {
                X3[j] = x[j] + h*(A31*F31[j] + A32*F32[j]);
            }

            /* Computing X4 */
            f(t+C4*h,X1,F41,fparams);
            f(t+C4*h,X2,F42,fparams);
            f(t+C4*h,X3,F43,fparams);
            for(j=0; j<nos; j++)
            {
                X4[j] = x[j] + h*(A41*F41[j] + A42*F42[j] + A43*F43[j]);
            }

            /* Computing X5 */
            f(t+C5*h,X1,F51,fparams);
            f(t+C5*h,X2,F52,fparams);
            f(t+C5*h,X3,F53,fparams);
            f(t+C5*h,X4,F54,fparams);
            for(j=0; j<nos; j++)
            {
                X5[j] = x[j] + h*(A51*F51[j] + A52*F52[j] + A53*F53[j] + A54*F54[j]);
            }

            /* Computing X6 */
            f(t+C6*h,X1,F61,fparams);
            f(t+C6*h,X2,F62,fparams);
            f(t+C6*h,X3,F63,fparams);
            f(t+C6*h,X4,F64,fparams);
            f(t+C6*h,X5,F65,fparams);
            for(j=0; j<nos; j++)
            {
                X6[j] = x[j] + h*(A61*F61[j] + A62*F62[j] + A63*F63[j] + A64*F64[j] + A65*F65[j]);
            }

            /* Computing X7 */
            f(t+C7*h,X1,F71,fparams);
            f(t+C7*h,X2,F72,fparams);
            f(t+C7*h,X3,F73,fparams);
            f(t+C7*h,X4,F74,fparams);
            f(t+C7*h,X5,F75,fparams);
            f(t+C7*h,X6,F76,fparams);
            for(j=0; j<nos; j++)
            {
                X7[j] = x[j] + h*(A71*F71[j] + A72*F72[j] + A73*F73[j] + A74*F74[j] + A75*F75[j] + A76*F76[j]);
            }


            /* Computing standard (x1) and more accurate (x2) solution */
            f(t+C1*h,X1,F11,fparams);
            f(t+C2*h,X2,F22,fparams);
            f(t+C3*h,X3,F33,fparams);
            f(t+C4*h,X4,F44,fparams);
            f(t+C5*h,X5,F55,fparams);
            f(t+C6*h,X6,F66,fparams);
            f(t+C7*h,X7,F77,fparams);
            /* double ETOL = 0; */
            double rel_err_max = 0;
            for(j=0; j<nos; j++)
            {
                x1[j] = x[j] + h*(B61*F11[j] + B62*F22[j] + B63*F33[j] + B64*F44[j] + B65*F55[j] + B66*F66[j]);
                x2[j] = x[j] + h*(B71*F11[j] + B72*F22[j] + B73*F33[j] + B74*F44[j] + B75*F55[j] + B76*F66[j] + B77*F77[j]);
                err[j] = x1[j]-x2[j];


                /* Compute local relative error */
                double r_err = 0;
                if (fabs(x1[j]) > THRESHOLD)
                {
                    if (fabs(x1[j]) > fabs((*state)[i]))
                    {
                        r_err = fabs(err[j]/x1[j]);
                    }
                    else
                    {
                        r_err = fabs(err[j]/(*state)[i]);
                    }
                }
                else
                {
                    if (fabs((*state)[i]) > THRESHOLD)
                    {
                        r_err = fabs(err[j]/(*state)[i]);
                    }
                    else
                    {
                        r_err = fabs(err[j]/THRESHOLD);
                    }
                }

                /* Store maximum local error */
                if (r_err > rel_err_max)
                {
                    rel_err_max = r_err;
                }

                /* Check solution tolerance */
                if (variable_step && r_err > FRAC*RTOL)
                {
                    reject_solution = true;
                }
            }

            (*time)[i+1] = t+h;

            /* Adapt step size */
            if (variable_step)
            {
                if (reject_solution)
                {
                    h = 0.5*h;
                }
                else
                {
                    /* h = h*pow(FRAC*ETOL/err_max,POW); */
                    h = h*pow(FRAC*RTOL/rel_err_max,POW);
                }
                if (h < hMIN)
                {
                    /* mexWarnMsgTxt("Minimum step size criterium not met."); */
                    h = hMIN;
                }
                if (h > hMAX)
                {
                    h = hMAX;
                }
            }
        }

        i++;
        t = (*time)[i];
        memcpy(x,x2,nos*sizeof(double));
        for(j=0; j<nos; j++)
        {
            (*state)[i*nos+j] = x1[j];
        }

    }

    if (variable_step)
    {
        *ndata = i+1;
    }

    free(x);
    free(X1);
    free(X2);
    free(X3);
    free(X4);
    free(X5);
    free(X6);
    free(X7);
    free(x1);
    free(x2);
    free(err);
    free(F21);
    free(F31);
    free(F32);
    free(F41);
    free(F42);
    free(F43);
    free(F51);
    free(F52);
    free(F53);
    free(F54);
    free(F61);
    free(F62);
    free(F63);
    free(F64);
    free(F65);
    free(F71);
    free(F72);
    free(F73);
    free(F74);
    free(F75);
    free(F76);
    free(F11);
    free(F22);
    free(F33);
    free(F44);
    free(F55);
    free(F66);
    free(F77);

    return;
}


#define ode45_ERROR -99
#define ode45_SUCCESS 0

int ode45(
    ode_function f, /* System dynamics */
    void *function_params, /* System parameters */
    double ti,			/* Initial time instant */
    double tf,			/* Final time instant */
    const double *x0,			/* Initial state vector */
    unsigned int state_dim,	/* State dimension */
    double **time,		/* Time signal (out)*/
    double **state,		/* State signal (out)*/
    unsigned int &ndata, /* Number of output time instants */
    double h /* Step size */
)
{
    /* ######## Local variable declaration ######## */

    unsigned int nos;		/* Number of states */
    bool variable_step;		/* Use variable step size or fixed step size */


    /* ######## Validate Input Data ######## */

    variable_step = h > 0.0 ? false : true;


    /* Check initial state dimension and values */
    nos = state_dim;
    unsigned int i;
    for(i = 0; i < nos; i++)
    {
        if (isnan(x0[i]) || isinf(x0[i]))
        {
            fprintf(stderr, "The initial state must be a vector of real numbers.\n");
            return ode45_ERROR;
        }
    }

    /* ######## Call C subroutine ######## */

    if (variable_step)
    {
        ndata = 1;
        h = 2*(tf-ti);
//        std::cerr << "variable step" << std::endl;
    }
    else
    {
        ndata = ((unsigned int) (tf/h)) + 1;
    }
    *time  = (double*) calloc(ndata, sizeof(double));
    *state = (double*) calloc(nos * ndata, sizeof(double));
    ode45_stepper(f,function_params,variable_step,h,ti,tf,x0,nos,time,state,&ndata);

    return ode45_SUCCESS;
}
