#include "cprnd_pure.h"
#include <ilcplex/ilocplex.h>
#include <cmath>
#include <cstdlib>
#include <ctime>

#ifdef CPP11
#include <random>
extern std::mt19937 gen;
#endif

using namespace std;

#define TWO_PI 6.2831853071795864769252866
/*http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform*/
double generateGaussianNoise(const double mu, const double sigma)
{
    using namespace std;
    static bool haveSpare = false;
    static double rand1, rand2;

    if(haveSpare)
    {
        haveSpare = false;
        return (sigma * sqrt(rand1) * sin(rand2)) + mu;
    }

    haveSpare = true;

    rand1 = rand() / static_cast<double>(RAND_MAX);
    if(rand1 < 1e-100) rand1 = 1e-100;
    rand1 = -2 * log(rand1);
    rand2 = (rand() / static_cast<double>(RAND_MAX)) * TWO_PI;

    return (sigma * sqrt(rand1) * cos(rand2)) + mu;
}


/* x = A*b */
void laprodmv(double *A, int m, int n, double* b, double* x)
{
    int c, d, idx;
    double sum;
    for ( c = 0 ; c < m ; ++c )
    {
        sum = 0.0;
        for ( d = 0 ; d < n ; ++d )
        {
            idx =  IDX(c,d,m);
            sum += A[idx]*b[d];
        }
        x[c] = sum;
    }
}
/* X = A*B */
void laprodmm(double *A, int m, int n, double* B, int p, double* X)
{
    int c, d, k;
    double sum = 0.0;
    for ( c = 0 ; c < m ; ++c )
    {
        for ( d = 0 ; d < p ; ++d )
        {
            for ( k = 0 ; k < n ; ++k )
            {
                sum = sum + A[IDX(c,k,m)]*B[IDX(k,d,n)];
            }

            X[IDX(c,d,m)] = sum;
            sum = 0.0;
        }
    }
}

/* X = v1 * v2' */
void laprodvv(double *v1, int m, double* v2, double *X)
{
    int c, d;
    for (c = 0; c < m; ++c) {
        for (d = 0; d < m; ++d) {
            X[IDX(c,d,m)] = v1[c]*v2[d];
        }
    }
}

void printMat(double *A, int m, int n) {

    int c, d;
    for (c = 0; c < m; ++c) {
        for (d = 0; d < n; ++d) {
            cerr << A[IDX(c,d,m)] << " ";
        }
        cerr << "\n";
    }
}

void gendir(int dim, bool orthogonal, double *v)
{
#ifdef CPP11
    std::normal_distribution<> d(0,1);
#endif
    if (orthogonal == false)
    {
        double tot = 0;
        for (int i = 0; i < dim; i++) {
#ifdef CPP11
            v[i] = d(gen);
#else
            v[i] = generateGaussianNoise(0,1);
#endif
            tot += v[i]*v[i];
        }
        tot = sqrt(tot);
        for (int i = 0; i < dim; i++) {
            v[i] /= tot;
        }
    }
    else
    {
        abort();
    }
}


void chebycenter(double* A, double* b, int m, int n, double *x, bool& feasible)
{
    double* an = new double[m];
    for (int i = 0; i < m; ++i)
    {
        double tot = 0;
        for (int j = 0; j < n; j++) {
            double val = A[IDX(i,j,m)];
            tot += val*val;
        }
        an[i] = sqrt(tot);
    }

    //CPLEX
    IloEnv env;
    IloModel model(env);
    IloNumVarArray var(env);
    IloRangeArray c(env);
    for (int i = 0; i < n+1; i++)
    {
        var.add(IloNumVar(env,-IloInfinity));
    }

    for (int i = 0; i < m; i++)
    {
        IloExpr lhs(env);
        for (int j = 0; j < n; j++)
        {
            lhs += A[IDX(i,j,m)]*var[j];
        }
        lhs += an[i]*var[n];
        c.add(lhs <= b[i]);
    }
    c.add(-var[n] <= 0);
    model.add(c);
    model.add(IloMinimize(env, -var[n]));
    IloCplex cplex(model);

    // Optimize the problem and obtain solution.
    if ( !cplex.solve() )
    {
        //        env.error() << "Failed to optimize LP" << endl;
        //        throw(-1);
        for (int i = 0; i < n; ++i)
            x[i] = NAN;

        feasible = false;
        env.end();
        return;
    }
    feasible = true;

    IloNumArray vals(env);
    cplex.getValues(vals, var);
#if 0
    env.out() << "Solution status = " << cplex.getStatus() << endl;
    env.out() << "Solution value  = " << cplex.getObjValue() << endl;
    env.out() << "Values        = " << vals << endl;
#endif


    for (int i = 0; i < n; ++i)
        x[i] = vals[i];

    //    cout <<  outvect.t() << endl;
    //    printMat(x, n, 1);

    env.end();

    cout << "Ended Chebyshev center" << endl;

    delete [] an;

    return;
}


void cprnd_pure(double *A, double *b, int m, int n, int nbPoints, double* points)
{


    double isotropic = 123.456;

    int N = nbPoints;
    bool orthogonal = false;
    Methods method = ACHR;


    int runup = -1;
    int discard = -1;

    double* x0 = NULL;


    if (m < n+1)
    {
        cerr << "error: at least " << n+1 << " inequalities required" << endl;
        abort();
    }

    if (isotropic == 123.456)
    {
        if (method != ACHR)
            isotropic = 2;
        else
            isotropic = 0;
    }

    // Choose a starting point x0 if user did not provide one.
    if (x0 == NULL)
    {
        x0 = new double[n];
        bool feasible;
        chebycenter(A, b, m, n, x0, feasible);
        if (!feasible)
        {
            throw(-1);
        }

        //        cout << "x0:\n";
        //        printMat(x0,n,1);
    }

    // Default the runup to something arbitrary.
    if (runup < 0)
    {
        if (method == ACHR)
            runup = 10*(n+1);
        else if (isotropic > 0)
            runup = 10*n*(n+1);
        else
            runup = 0;
    }

    // Default the discard to something arbitrary
    if (discard < 0)
    {
        if (method == ACHR)
            discard = 25*(n+1);
        else
            discard = runup;
    }

    cout << "isotropic:" << isotropic << endl;
    cout << "runup:" << runup << endl;
    cout << "discard:" << discard << endl;


    int nbTempPoints = N+runup+discard;
    int i,j,idx,offset,k = 0;
    double *x = x0;
    double tmin, tmax, rnd;


    double* X  = new double[nbTempPoints*n];
    double* M  = new double[n]; //Incremental mean
    double* u  = new double[n]; //normalized direction
    double* z  = new double[m];
    double* ct = new double[m];
    double* c  = new double[m];

    double* delta0  = new double[n];


    for (i = 0; i < n; ++i) {
        M[i] = 0.0;
    }

    while (k < nbTempPoints)
    {
        assert(isotropic == 0);
        //////////////// ACHR ///////////////
        if (k < runup) {
            gendir(n,orthogonal,u);
        } else {
#ifdef CPP11
            std::uniform_int_distribution<> dis(0, k-1);
            int row_id = dis(gen);
#else
            int row_id = rand() % k;
#endif
            double norm2val = 0.0;
            for (i=0; i < n; ++i) {
                idx = IDX(row_id,i,nbTempPoints);
                u[i] = X[idx] - M[i];
                norm2val += u[i]*u[i];
            }
            norm2val = sqrt(norm2val);
            for (i=0; i < n; ++i) {
                u[i] = u[i] / norm2val;
            }
        }
        laprodmv(A,m,n,u,z);
        laprodmv(A,m,n,x,ct);
        for (i = 0; i < m; ++i) {
            c[i] = (b[i] - ct[i]) / z[i];
        }

        tmin = -1e8;
        tmax =  1e8;
        for (i = 0; i < m; ++i) {
            if ((z[i] < 0.0) && (c[i] > tmin))
                tmin = c[i];
            if ((z[i] > 0.0) && (c[i] < tmax))
                tmax = c[i];
        }

#ifdef CPP11
        rnd = std::generate_canonical<double, 10>(gen);
#else
        rnd = rand() / static_cast<double>(RAND_MAX);
#endif
        for (i=0; i < n; ++i) {
            x[i] += (tmin+(tmax-tmin)*rnd)*u[i];

            idx = IDX(k,i,nbTempPoints);
            X[idx] = x[i];
            delta0[i] = x[i] - M[i];
            M[i] += delta0[i] / (k+1);
        }

        ++k;

    }

    //    printMat(X,nbTempPoints,n);

    offset = discard+runup;
    for (i = 0; i < N; ++i) {
        for (j = 0; j < n; ++j) {
            idx = IDX(i+offset,j,nbTempPoints);
            points[IDX(i,j,N)] = X[idx];
        }
    }


    delete [] X;
    delete [] x0;
    delete [] M;
    delete [] u;
    delete [] z;
    delete [] ct;
    delete [] c;
    delete [] delta0;

}
