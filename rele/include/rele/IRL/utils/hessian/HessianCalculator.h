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

#ifndef INCLUDE_RELE_IRL_UTILS_HESSIANCALCULATOR_H_
#define INCLUDE_RELE_IRL_UTILS_HESSIANCALCULATOR_H_

#include "rele/approximators/BasisFunctions.h"
#include "rele/policy/Policy.h"
#include "rele/core/Transition.h"

namespace ReLe
{

template<class ActionC, class StateC>
class HessianCalculator
{
public:
    HessianCalculator(Features& phi,
                      Dataset<ActionC,StateC>& data,
                      DifferentiablePolicy<ActionC,StateC>& policy,
                      double gamma)
    {
        computeHessianDiff(phi, data, policy, gamma);
    }

    arma::mat computeHessian(const arma::vec& w)
    {
        arma::mat H(Hdiff.n_rows, Hdiff.n_cols, arma::fill::zeros);

        for(unsigned int i = 0; i < Hdiff.n_slices; i++)
        {
            H += Hdiff.slice(i)*w(i);
        }

        return H;

    }

    arma::cube getHessianDiff()
    {
        return Hdiff;
    }

private:
    void computeHessianDiff(Features& phi,
                            Dataset<ActionC,StateC>& data,
                            DifferentiablePolicy<ActionC,StateC>& policy,
                            double gamma)
    {
        unsigned int parameterSize = policy.getParametersSize();
        Hdiff.zeros(parameterSize, parameterSize, phi.rows());

        for(unsigned int ep = 0; ep < data.getEpisodesNumber(); ep++)
        {
            Episode<ActionC,StateC>& episode = data[ep];
            double df = 1.0;

            for (unsigned int t = 0; t < episode.size(); t++)
            {
                Transition<ActionC,StateC>& tr = episode[t];
                arma::mat K = computeK(policy, tr);

                arma::vec phi_t = phi(tr.x, tr.u, tr.xn);

                for(unsigned int f = 0; f < phi.rows(); f++)
                {
                    Hdiff.slice(f) += df*K*phi_t(f);
                }

                df *= gamma;

            }
        }

    }

    arma::mat computeK(DifferentiablePolicy<ActionC,StateC>& policy,
                       Transition<ActionC,StateC>& tr)
    {
        arma::vec logDiff = policy.difflog(tr.x, tr.xn);
        arma::mat logDiff2 = policy.diff2log(tr.x, tr.xn);
        return logDiff2 + logDiff*logDiff.t();
    }


private:
    arma::cube Hdiff;
};

}


#endif /* INCLUDE_RELE_IRL_UTILS_HESSIANCALCULATOR_H_ */
