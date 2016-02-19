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

#ifndef INCLUDE_RELE_IRL_OPTIMIZATION_SIMPLEX_H_
#define INCLUDE_RELE_IRL_OPTIMIZATION_SIMPLEX_H_

#include <armadillo>
#include <vector>

namespace ReLe
{

class Simplex
{
public:
    Simplex(unsigned int size) : size(size)
    {
        // all features are active by default
        active_feat.set_size(size);
        std::iota(std::begin(active_feat), std::end(active_feat), 0);
        compute();
    }

    inline arma::vec reconstruct(const arma::vec& xSimplex)
    {
        int dim = active_feat.n_elem;

        arma::vec x(size, arma::fill::zeros);

        x(active_feat(arma::span(0, dim - 2))) = xSimplex;
        x(active_feat(dim - 1)) = 1.0 - sum(xSimplex);

        return x;
    }


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

    inline arma::vec reconstruct()
    {
        arma::vec x(size, arma::fill::zeros);
        x.elem(active_feat).ones();

        return x;
    }

    inline arma::mat diff(const arma::mat& df)
    {
        return dtheta_simplex * df;
    }

    inline unsigned int getEffectiveDim()
    {
        return active_feat.n_elem - 1;
    }

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

#endif /* INCLUDE_RELE_IRL_OPTIMIZATION_SIMPLEX_H_ */
