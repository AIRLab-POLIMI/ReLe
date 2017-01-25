/*
 * rele,
 *
 *
 * Copyright (C) 2017 Davide Tateo
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

#include "rele/policy/utils/LinearStatisticEstimation.h"

#include <cassert>

namespace ReLe
{

void LinearStatisticEstimation::computeMLE(const Features& phi, const Dataset<DenseAction,DenseState>& data)
{
    assert(data.size() > 0 && data[0].size() > 0);
    assert(phi.cols() == 1);

    unsigned int uDim = data[0][0].u.n_elem;

    arma::mat X(phi.rows(), data.getTransitionsNumber());
    arma::mat U(uDim, data.getTransitionsNumber());

    unsigned int index = 0;
    for(auto& episode : data)
    {
        for(auto& tr : episode)
        {
            X.col(index) = phi(tr.x);
            U.col(index) = tr.u;

            index++;
        }
    }

    arma::mat W = U*arma::pinv(X);
    Sigma = arma::zeros(uDim, uDim);

    unsigned int n = 0;
    for(auto& episode : data)
    {
        for(auto& tr : episode)
        {
            arma::vec xn = X.col(n);
            arma::vec un = U.col(n);
            arma::vec delta = un - W*xn;
            Sigma += delta*delta.t();

            n++;
        }
    }

    Sigma /= n;
    theta =  arma::vectorise(W.t());

}

arma::vec LinearStatisticEstimation::getMeanParameters()
{
    return theta;
}

arma::mat LinearStatisticEstimation::getCovariance()
{
    return Sigma;
}

}
