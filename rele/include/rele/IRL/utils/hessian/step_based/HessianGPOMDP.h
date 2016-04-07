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

#ifndef INCLUDE_RELE_IRL_UTILS_HESSIAN_HESSIANGPOMDP_H_
#define INCLUDE_RELE_IRL_UTILS_HESSIAN_HESSIANGPOMDP_H_

#include "rele/IRL/utils/hessian/step_based/StepBasedHessianCalculator.h"

namespace ReLe
{
template<class ActionC, class StateC>
class HessianGPOMDP : public StepBasedHessianCalculator<ActionC, StateC>
{
protected:
    USING_STEP_BASED_H_CALCULATORS_MEMBERS(ActionC, StateC)

public:
    HessianGPOMDP(Features& phi,
                  Dataset<ActionC,StateC>& data,
                  DifferentiablePolicy<ActionC,StateC>& policy,
                  double gamma) : StepBasedHessianCalculator<ActionC, StateC>(phi, data, policy, gamma)
    {

    }

    virtual ~HessianGPOMDP()
    {

    }

protected:
    virtual arma::cube computeHessianDiff() override
    {
        unsigned int dp = policy.getParametersSize();
        unsigned int dr = phi.rows();
        unsigned int episodeN = data.size();

        arma::cube Hdiff(dp, dp, phi.rows(), arma::fill::zeros);

        for(auto& episode : data)
        {
            //core setup
            arma::vec sumGradLog(dp, arma::fill::zeros);
            arma::mat sumHessLog(dp, dp, arma::fill::zeros);
            double df = 1.0;

            for(auto& tr : episode)
            {
                sumGradLog += policy.difflog(tr.x, tr.u);
                sumHessLog += policy.diff2log(tr.x, tr.u);
                arma::vec creward = phi(tr.x, tr.u, tr.xn);

                arma::mat G = sumGradLog*sumGradLog.t() + sumHessLog;

                // compute the gradients
                for(unsigned int r = 0; r < dr; r++)
                    Hdiff.slice(r) += df*creward(r)*G;

                df *= gamma;
            }

        }

        Hdiff /= episodeN;

        return Hdiff;
    }



};

template<class ActionC, class StateC>
class HessianGPOMDPBase : public StepBasedHessianCalculator<ActionC, StateC>
{
protected:
    USING_STEP_BASED_H_CALCULATORS_MEMBERS(ActionC, StateC)

public:
    HessianGPOMDPBase(Features& phi,
                      Dataset<ActionC,StateC>& data,
                      DifferentiablePolicy<ActionC,StateC>& policy,
                      double gamma) : StepBasedHessianCalculator<ActionC, StateC>(phi, data, policy, gamma)
    {

    }

    virtual ~HessianGPOMDPBase()
    {

    }

protected:
    virtual arma::cube computeHessianDiff() override
    {
        //TODO [IMPORTANT] implement or remove class
        return arma::cube();
    }
};

}


#endif /* INCLUDE_RELE_IRL_UTILS_HESSIAN_HESSIANGPOMDP_H_ */
