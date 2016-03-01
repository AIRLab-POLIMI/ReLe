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

#ifndef INCLUDE_RELE_IRL_UTILS_HESSIAN_NONLINEAR_NONLINEARHESSIANGPOMDP_H_
#define INCLUDE_RELE_IRL_UTILS_HESSIAN_NONLINEAR_NONLINEARHESSIANGPOMDP_H_

#include "rele/IRL/utils/hessian_nonlinear/NonlinearHessianCalculator.h"

namespace ReLe
{

template<class ActionC, class StateC>
class NonlinearHessianGPOMDP : public NonlinearHessianCalculator<ActionC, StateC>
{
public:
    NonlinearHessianGPOMDP(ParametricRegressor& rewardFunc,
                           Dataset<ActionC, StateC>& data,
                           DifferentiablePolicy<ActionC, StateC>& policy, double gamma) :
        NonlinearHessianCalculator<ActionC, StateC>(rewardFunc, data, policy, gamma)
    {

    }

    virtual void compute() override
    {
        unsigned int dp = this->policy.getParametersSize();
        unsigned int dr = this->rewardFunc.getParametersSize();
        this->H.zeros(dp, dp);
        this->Hdiff.zeros(dp, dp, dr);

        for (auto& episode : this->data)
        {
            double df = 1.0;
            arma::vec sumGradLog(dp, arma::fill::zeros);
            arma::mat sumHessLog(dp, dp, arma::fill::zeros);

            for (auto& tr : episode)
            {
                double creward = arma::as_scalar(this->rewardFunc(tr.x, tr.u, tr.xn));
                arma::vec dcreward = this->rewardFunc.diff(tr.x, tr.u, tr.xn);

                sumGradLog += this->policy.difflog(tr.x, tr.u);
                sumHessLog += this->policy.diff2log(tr.x, tr.u);

                arma::mat G = sumGradLog*sumGradLog.t()+sumHessLog;

                this->H += df*creward*G;

                for(unsigned int r = 0; r < dr; r++)
                    this->Hdiff.slice(r) += df*dcreward(r)*G;

                df *= this->gamma;
            }
        }

        this->H /= this->data.size();
        this->Hdiff /= this->data.size();

    }

    virtual ~NonlinearHessianGPOMDP()
    {

    }
};


template<class ActionC, class StateC>
class NonlinearHessianGPOMDPBase : public NonlinearHessianCalculator<ActionC, StateC>
{
public:
    NonlinearHessianGPOMDPBase(ParametricRegressor& rewardFunc,
                               Dataset<ActionC, StateC>& data,
                               DifferentiablePolicy<ActionC, StateC>& policy, double gamma) :
        NonlinearHessianCalculator<ActionC, StateC>(rewardFunc, data, policy, gamma)
    {

    }

    virtual void compute() override
    {

    }

    virtual ~NonlinearHessianGPOMDPBase()
    {

    }
};

}


#endif /* INCLUDE_RELE_IRL_UTILS_HESSIAN_NONLINEAR_NONLINEARHESSIANGPOMDP_H_ */
