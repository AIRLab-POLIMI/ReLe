/*
 * C file for ODE integration (Dorman-Prince RK45)
 * Alessandro Falsone and Matteo Pirotta
 * Politecnico di MIlano - 2014
 *
 * Last changes:
 *  [name, date, change]
 */

#include <iostream>

/**
 * This function should store the vector elements f_i(t,y,params) in the array dydt, for arguments (t,y) and parameters params.
 */
typedef int (* ode_function) (double t, const double y[], double dydt[], void * params);

/**
 * @brief Dorman-Prince RK45 stepper function
 * @param f system dynamics
 * @param variable_step use variable step
 * @param h step size
 * @param ti initial time instant
 * @param tf final time instant
 * @param x0 initial state vector
 * @param nos state dimension
 * @param time time signal containing the integration history
 * @param state state signal containing the integration history
 * @param ndata
 */
void ode45_stepper(
    ode_function f,
    void* fparams,
    bool variable_step,
    double h,
    double ti,
    double tf,
    const double *x0,
    unsigned int nos,
    double **time,
    double **state,
    unsigned int *ndata);

/**
 * @brief ode45
 * @param ti
 * @param tf
 * @param x0
 * @param state_dim
 * @param time
 * @param state
 * @param h
 */
int ode45(ode_function f, /* System dynamics */
          void *function_params, /* System parameters */
          double ti,			/* Initial time instant */
          double tf,			/* Final time instant */
          const double *x0,			/* Initial state vector */
          unsigned int state_dim,	/* State dimension */
          double **time,		/* Time signal (out)*/
          double **state,		/* State signal (out)*/
          unsigned int &ndata,
          double h = -1.0		/* Step size */
         );

