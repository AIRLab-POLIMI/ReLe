/*
 * rele,
 *
 *
 * Copyright (C) 2015 Davide Tateo & Matteo Pirotta
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

#ifndef INCLUDE_RELE_UTILS_UTILS_H_
#define INCLUDE_RELE_UTILS_UTILS_H_

#include <cmath>
#include <armadillo>

#include <functional>


namespace ReLe
{

class utils
{
public:

    /**
     * @brief Cholesky safe decomposition using nearestSPD
     */
    static inline arma::mat chol(arma::mat& M)
    {
        if(M.n_elem == 1 && M(0) <= 0)
        {
            arma::mat C(1, 1, arma::fill::randn);
            C(0) = std::numeric_limits<double>::epsilon();
            return C;
        }

        try
        {
            return arma::chol(M);
        }
        catch(std::runtime_error& e)
        {
            arma::mat B = (M + M.t())/2;

            arma::mat U, V;
            arma::vec s;
            bool ok = arma::svd(U, s, V, M);

            if(!ok)
                throw std::runtime_error("Bad Covariance Matrix, SVD failed");

            arma::mat H = V*arma::diagmat(s)*V.t();

            arma::mat C = (B+H)/2;
            C = (C + C.t())/2;

            int k = 0;
            while(true)
            {
                try
                {
                    return arma::chol(C);
                }
                catch(std::runtime_error& e)
                {
                    k++;
                    double k2 = k*k;
                    double mineig = arma::min(arma::eig_sym(C));
                    double relEps = std::nextafter(mineig, std::numeric_limits<double>::infinity()) - mineig;
                    C += (-mineig*k2 + relEps)*arma::eye(C.n_rows, C.n_cols);
                }
            }

        }

    }

    /**
     * @brief Numerical gradient computation
     */
    template<class F>
    static inline arma::vec computeNumericalGradient(F J, arma::vec theta)
    {

        arma::vec numgrad = arma::zeros(theta.n_elem);
        arma::vec perturb = arma::zeros(theta.n_elem);
        double e = 1e-5;

        for (unsigned int p = 0; p < theta.n_elem; p++)
        {
            // Set perturbation vector
            perturb(p) = e;
            double loss1 = J(theta - perturb);
            double loss2 = J(theta + perturb);
            // Compute Numerical Gradient
            numgrad(p) = (loss2 - loss1) / (2*e);
            perturb(p) = 0;
        }

        return numgrad;

    }

};

}
#endif /* INCLUDE_RELE_UTILS_UTILS_H_ */
