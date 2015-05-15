#include "ArmadilloExtensions.h"

namespace ReLe
{

arma::mat null(const arma::mat &A, double tol)
{
    int m = A.n_rows;
    int n = A.n_cols;

    arma::mat U, V;
    arma::vec s;
    if (m <= n)
    {
        arma::svd(U, s, V, A);
    }
    else
    {
        arma::svd_econ(U, s, V, A);
    }

    // U.save("/tmp/ReLe/lqr/GIRL/U.log", arma::raw_ascii);
    // s.save("/tmp/ReLe/lqr/GIRL/s.log", arma::raw_ascii);
    // V.save("/tmp/ReLe/lqr/GIRL/V.log", arma::raw_ascii);

    if (tol == -1)
    {
        tol = std::max(m,n) * max(s) * 2.220446049250313e-16; //look at matlab implementation
    }
    arma::uvec tmp = arma::find(s > tol);
    unsigned int r = tmp.n_elem;
    return V.cols(r, n-1);
}


arma::uvec rref(const arma::mat& X, arma::mat& A, double tol)
{
    int m = X.n_rows;
    int n = X.n_cols;

    if (tol == -1)
    {
        tol = std::max(m,n) * 2.220446049250313e-16 * arma::norm(X,"inf");
    }

    unsigned int i = 0, j = 0;
    arma::uvec jb;
    A = X;
    while ((i < m) && (j < n))
    {
        //Find value and index of largest element in the remainder of column j.
        arma::vec tmp = A(arma::span(i,m-1), arma::span(j,j));
        tmp = abs(tmp);
        arma::uword k;
        double p = tmp.max(k);
        k = k + i;
        if (p <= tol)
        {
            //The column is negligible, zero it out.
            A(arma::span(i,m-1), arma::span(j,j)).zeros();
            ++j;
        }
        else
        {
            //Remember column index
            arma::uvec u = {j};
            jb = arma::join_vert(jb,u);

            // Swap i-th and k-th rows.
            arma::uvec idx = {i,k};
            arma::uvec idx2 = {k,i};
            arma::uvec aaa(n-j);
            int ii = 0;
            for (int u = j; u < n; ++u)
                aaa(ii++) = u;
            A.submat(idx,aaa) = A.submat(idx2,aaa);

            //Divide the pivot row by the pivot element.
            A(arma::span(i,i),arma::span(j,n-1)) = A(arma::span(i,i),arma::span(j,n-1))/A(i,j);

            //Subtract multiples of the pivot row from all the other rows.
            std::vector<int> vv;
            for (int u = 0, ue = i; u < ue; ++u)
                vv.push_back(u);
            for (int u = i+1, ue = m; u < ue; ++u)
                vv.push_back(u);

            for (auto k : vv)
            {
                A(k,arma::span(j,n-1)) = A(k,arma::span(j,n-1)) - A(k,j)*A(i,arma::span(j,n-1));
            }

            i = i + 1;
            j = j + 1;
        }
    }
    return jb;
}

double wrapTo2Pi(double lambda)
{
    double val = fmod(lambda, 2.0*M_PI);
    if ((lambda > 0) && (val == 0.0))
    {
        return 2*M_PI;
    }
    return val;
}

arma::vec wrapTo2Pi(const arma::vec& lambda)
{
    unsigned int n = lambda.n_elem;
    arma::vec tmp(n);
    for (unsigned int i = 0; i < n; ++i)
    {
        tmp(i) = wrapTo2Pi(lambda(i));
    }
    return tmp;
}

double wrapToPi(double lambda)
{
    if ((lambda < -M_PI) || (M_PI < lambda))
    {
        return wrapTo2Pi(lambda + M_PI) - M_PI;
    }
    return lambda;
}

arma::vec wrapToPi(const arma::vec& lambda)
{
    unsigned int n = lambda.n_elem;
    arma::vec ret(n);
    for (unsigned int i = 0; i < n; ++i)
    {
        ret(i) = wrapTo2Pi(lambda(i) + M_PI) - M_PI;
    }
    return ret;
}

//    arma::uvec licols(const arma::mat& X, arma::mat& Xsub, double tol = 1e-10)
//    {
//        arma::uvec idx;
//        arma::uvec a = arma::find(X==0);
//        if (a.n_elem == X.n_elem)
//        {
//            Xsub.set_size(0,0);
//            return idx;
//        }

//        int m = X.n_rows;
//        int n = X.n_cols;

//        arma::mat Q, R;
//        if (m <= n)
//        {
//            arma::qr(Q,R,X);
//        }
//        else
//        {
//            arma::qr_econ( Q, R, X );
//        }

//        arma::vec diagr;
////        //        if ((R.n_rows > 1) && (R.n_cols > 1))
////        //        {
//        diagr = abs(R.diag());
////        //        }
////        //        else
////        //        {
////        //            diagr = R(0);
////        //        }

//        std::cerr << R;

//        arma::vec absdiag = abs(diagr);
//        arma::uvec E = arma::sort_index(absdiag,"descend");


//        //rank estimation
//        arma::uvec rv = arma::find(diagr >= tol*diagr(0)); //rank estimation
//        int r = rv.n_elem;

//        arma::uvec e = E.rows(0,r-1);
//        idx = arma::sort(e);

//        Xsub = X.cols(idx);
//        std::cerr << Xsub;

//        return idx;
//    }

}
