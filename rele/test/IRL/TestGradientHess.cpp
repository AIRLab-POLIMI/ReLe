/*
 * rele,
 *
 *
 * Copyright (C) 2016 Davide Tateo
 * Versione 1.0
 *
 * This file is part of rele.
 *
 * rele is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * rele is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with rele.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <armadillo>

#include "rele/utils/NumericalGradient.h"

arma::cube diff;

arma::mat computeHessian(const arma::vec& w)
{
    arma::mat H(diff.n_rows, diff.n_cols, arma::fill::zeros);

    for(unsigned int i = 0; i < w.n_rows; i++)
    {
        H += w(i)*diff.slice(i);
    }

    H += (1.0 - arma::sum(w))*diff.slice(w.n_rows);

    return H;
}

double maxEigenvalue(const arma::vec& w)
{
    arma::mat hessian = computeHessian(w);
    arma::vec lambda = arma::eig_sym(hessian);
    return arma::as_scalar(lambda.tail(1));
}

arma::vec maxEigenvalueDiff(const arma::vec& w)
{
    arma::mat hessian = computeHessian(w);
    arma::vec grad(w.n_rows);

    arma::mat diff_n = diff.slice(w.n_elem);

    arma::vec lambda;
    arma::mat V;

    arma::eig_sym(lambda, V, hessian);

    arma::vec vi = V.tail_cols(1);


    for(unsigned int i = 0; i < w.n_elem; i++)
    {
        arma::mat diff_i = diff.slice(i);
        grad(i) = arma::as_scalar(vi.t()*(diff_i-diff_n)*vi);
    }

    return grad;

}

int main(int argc, char *argv[])
{
    unsigned int dp = 3;
    unsigned int n = 3;
    diff = arma::cube(dp, dp, n);

    arma::vec wComp(n, arma::fill::randn);
    wComp /= arma::sum(wComp);
    arma::vec w = wComp.head_rows(n-1);

    for(unsigned int i = 0; i < n; i++)
    {
        arma::mat tmp = arma::randn(dp, dp);
        diff.slice(i) = (tmp+tmp.t())/2.0;
    }

    auto lambdaFunc = [](const arma::vec& par)
    {
        arma::vec p(1);
        p(0) = maxEigenvalue(par);
        return p;
    };

    arma::vec exact = maxEigenvalueDiff(w);
    arma::vec numerical = ReLe::NumericalGradient::compute(lambdaFunc, w);
    arma::vec delta = exact - numerical;

    std::cout << "lambda: " << maxEigenvalue(w) << std::endl;
    std::cout << "diff: " << exact.t() << std::endl;
    std::cout << "numerical: " << numerical.t() << std::endl;
    std::cout << "delta: " << delta.t() << std::endl;
}
