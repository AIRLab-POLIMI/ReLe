#ifndef CPRND_PURE_H
#define CPRND_PURE_H

enum Methods
{
    ACHR, GIBBS, HITANDRUN
};


#define IDX(i,j,nrow) (i) + (nrow)*(j)

double generateGaussianNoise(const double mu, const double sigma);

/* x = A*b */
void laprodmv(double *A, int m, int n, double* b, double* x);

/* X = A*B */
void laprodmm(double *A, int m, int n, double* B, int p, double* X);

/* X = v1 * v2' */
void laprodvv(double *v1, int m, double* v2, double *X);

void printMat(double *A, int m, int n);

void gendir(int dim, bool orthogonal, double *v);

void chebycenter(double* A, double* b, int m, int n, double *x, bool& feasible);

void cprnd_pure(double* A, double* b, int m, int n, int nbPoints, double *points);

#endif // CPRND_PURE_H

