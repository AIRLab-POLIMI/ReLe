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

#ifndef INCLUDE_RELE_IRL_UTILS_NONLINEARHESSIANCALCULATOR_H_
#define INCLUDE_RELE_IRL_UTILS_NONLINEARHESSIANCALCULATOR_H_

#include "rele/approximators/Regressors.h"
#include "rele/policy/Policy.h"
#include "rele/core/Transition.h"

namespace ReLe
{

template<class ActionC, class StateC>
class NonlinearHessianCalculator
{
public:
    NonlinearHessianCalculator(ParametricRegressor& rewardFunc,
                               Dataset<ActionC, StateC>& data,
                               DifferentiablePolicy<ActionC, StateC>& policy, double gamma) :
        rewardFunc(rewardFunc), data(data), policy(policy), gamma(gamma)
    {

    }

    virtual void compute() = 0;

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

protected:
    void computeEpisodeStatistics(Episode<ActionC,StateC>& episode, double& Rew,
                                  arma::vec& dRew, arma::vec& sumGradLog, arma::mat& sumHessLog)
    {
        double df = 1.0;
        for (auto& tr : episode)
        {
            Rew += df*arma::as_scalar(this->rewardFunc(tr.x, tr.u, tr.xn));;
            dRew += df*this->rewardFunc.diff(tr.x, tr.u, tr.xn);
            sumGradLog += this->policy.difflog(tr.x, tr.u);
            sumHessLog += this->policy.diff2log(tr.x, tr.u);

            df *= this->gamma;
        }
    }


protected:
    ParametricRegressor& rewardFunc;
    Dataset<ActionC, StateC>& data;
    DifferentiablePolicy<ActionC, StateC>& policy;
    double gamma;

    arma::mat H;
    arma::cube Hdiff;

};

}

#endif /* INCLUDE_RELE_IRL_UTILS_NONLINEARHESSIANCALCULATOR_H_ */
