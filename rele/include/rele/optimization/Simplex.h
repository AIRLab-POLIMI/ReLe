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

#ifndef INCLUDE_RELE_OPTIMIZATION_SIMPLEX_H_
#define INCLUDE_RELE_OPTIMIZATION_SIMPLEX_H_

#include <armadillo>
#include <vector>

namespace ReLe
{

/*!
 * This class implements the simplex one sum constraint for optimization.
 * Given a parameter space \f$\theta_{simplex}\in\mathbb{R}^{n-1}\f$ this class reconstructs
 * the full parametrization \f$\theta\in\mathbb{R}^n\f$ by appling the one sum constraint.
 * This class can also compute the derivative of the reduced parametrization.
 * Also, this class support non active parameters in the parameters vector, in order to compute the simplex
 * constraint on a subset of the original parametrization.
 */
class Simplex
{
public:
    /*!
     * Constructor.
     * \param size the dimension of the full parameter space
     */
    Simplex(unsigned int size) : size(size)
    {
        // all features are active by default
        active_feat.set_size(size);
        std::iota(std::begin(active_feat), std::end(active_feat), 0);
        compute();
    }

    /*!
     * Given a set of parameters in the simplex, reconstruct the full parametrization
     * \param xSimplex an armadillo vector of the simplex parameters
     * \return a vector to the full parametrization
     */
    inline arma::vec reconstruct(const arma::vec& xSimplex)
    {
        int dim = active_feat.n_elem;

        arma::vec x(size, arma::fill::zeros);

        x(active_feat(arma::span(0, dim - 2))) = xSimplex;
        x(active_feat(dim - 1)) = 1.0 - sum(xSimplex);

        return x;
    }

    /*!
     * Given a set of parameters in the simplex, reconstruct the full parametrization
     * \param xSimplex an std::vector of the simplex parameters
     * \return a vector to the full parametrization
     */
    inline arma::vec reconstruct(const std::vector<double> xSimplex)
    {
        // reconstruct parameters
        int dim = active_feat.n_elem;
        int n = xSimplex.size();

        arma::vec x(size, arma::fill::zeros);

        if (n == dim - 1)
        {
            // simplex scenario
            double sumx = 0.0;
            for (int i = 0; i < n; ++i)
            {
                x(active_feat(i)) = xSimplex[i];
                sumx += xSimplex[i];
            }
            x(active_feat(n)) = 1 - sumx;
        }

        return x;
    }

    /*!
     * returns the center of the simplex.
     * \return a vector to the full parametrization
     */
    inline arma::vec reconstruct()
    {
        arma::vec x(size, arma::fill::zeros);
        x.elem(active_feat).ones();
        x /= active_feat.n_elem;

        return x;
    }

    /*!
     * Compute the derivative of the function w.r.t. the simplex
     * \param df the derivative w.r.t. the full parametrization
     * \return the composite derivative matrix
     */
    inline arma::mat diff(const arma::mat& df)
    {
        return dtheta_simplex * df;
    }

    /*!
     * Getter.
     * \return the simplex dimension (considering only active features)
     */
    inline unsigned int getEffectiveDim()
    {
        return active_feat.n_elem - 1;
    }

    /*!
     * Setter.
     * \param active the active features
     */
    inline void setActiveFeatures(arma::uvec& active)
    {
        active_feat = active;
        compute();
    }

private:
    inline void compute()
    {
        unsigned int dim = active_feat.n_elem;
        unsigned int n = dim - 1;

        dtheta_simplex = arma::mat(n, size, arma::fill::zeros);

        int i;
        for (i = 0; i < n; i++)
        {
            unsigned int index = active_feat(i);
            dtheta_simplex(i, index) = 1.0;
        }

        dtheta_simplex.col(active_feat(i)) -= arma::ones(n);
    }

private:
    arma::uvec active_feat;
    arma::mat dtheta_simplex;

    unsigned int size;
};


}

#endif /* INCLUDE_RELE_OPTIMIZATION_SIMPLEX_H_ */
