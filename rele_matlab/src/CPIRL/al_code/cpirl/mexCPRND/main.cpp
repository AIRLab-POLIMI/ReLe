#include <iostream>
#include <armadillo>
#include "cprnd_pure.h"
#include <fstream>
#include <ctime>

using namespace std;
using namespace arma;

#ifdef CPP11
std::mt19937 gen(time(0));
#endif

int main(int argc, char* argv[])
{

    srand(time(0));

    int N = atoi(argv[3]); //number of points to be extracted

    //    mat A_mat(A, m, m, false);
    //    vec b_vec(b, m, false);

    mat A;
    vec b;
    A.load(argv[1]);
    b.load(argv[2]);

    //    cout << A << endl;
    //    cout << endl << b << endl;

    ofstream log("cprnd_log.xt", ios_base::out);

    //    mat points;
    double* points = new double[N*A.n_cols];
    try
    {
        cprnd_pure(A.memptr(), b.memptr(), A.n_rows, A.n_cols, N, points);

        //            for (unsigned int i = 0; i < N*A.n_cols; ++i)
        //            cout << points[i] << " ";
        printMat(points,N,A.n_cols);
    }
    catch (int e)
    {
        if (e == -1)
        {
            arma::vec media(A.n_cols, fill::none);
            for (unsigned int i = 0; i < A.n_cols; ++i)
            {
                media[i] = datum::nan;
            }
            media.save(argv[4], raw_ascii);
            log << "Infeasible";
            log.close();
        }
        return 1;
    }

    log << "Feasible";
    log.close();

    //    mat ptmat(points, N, A.n_cols);
    //    cout << ptmat << endl;


    delete [] points;

    return 0;
}
