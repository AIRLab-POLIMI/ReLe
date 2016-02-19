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

//TODO implement
template<class ActionC, class StateC>
class NonlinearHessianCalculator
{
public:
    NonlinearHessianCalculator(Regressor& rewardFunc,
                               Dataset<ActionC, StateC>& data,
                               DifferentiablePolicy<ActionC, StateC>& policy, double gamma) :
        rewardFunc(rewardFunc), data(data), policy(policy), gamma(gamma)
    {

    }

    virtual void compute(bool computeDerivative = true)
    {
        unsigned int dp = policy.getParametersSize();
        H.zeros(dp, dp);

        for (unsigned int ep = 0; ep < data.getEpisodesNumber(); ep++)
        {
            Episode<ActionC, StateC>& episode = data[ep];
            double df = 1.0;

            for (unsigned int t = 0; t < episode.size(); t++)
            {
                Transition<ActionC, StateC>& tr = episode[t];
                arma::mat K = computeK(policy, tr);

                double r = rewardFunc(tr.x, tr.u, tr.xn);

                H += df*K*r;

                df *= gamma;
            }
        }
    }

    arma::mat getHessian()
    {
        return H;
    }

    arma::cube getHessianDiff()
    {
        return Hdiff;
    }

    virtual ~NonlinearHessianCalculator()
    {

    }

private:
    //TODO in common class
    arma::mat computeK(DifferentiablePolicy<ActionC, StateC>& policy,
                       Transition<ActionC, StateC>& tr)
    {
        arma::vec logDiff = policy.difflog(tr.x, tr.xn);
        arma::mat logDiff2 = policy.diff2log(tr.x, tr.xn);
        return logDiff2 + logDiff * logDiff.t();
    }

private:
    Regressor& rewardFunc;
    Dataset<ActionC, StateC>& data;
    DifferentiablePolicy<ActionC, StateC>& policy;
    double gamma;

    arma::mat H;
    arma::cube Hdiff;

};

}

#endif /* INCLUDE_RELE_IRL_UTILS_HESSIANCALCULATOR_H_ */
