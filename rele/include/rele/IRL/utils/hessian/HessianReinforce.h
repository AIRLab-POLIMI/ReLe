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

#ifndef INCLUDE_RELE_IRL_UTILS_HESSIAN_HESSIANREINFORCE_H_
#define INCLUDE_RELE_IRL_UTILS_HESSIAN_HESSIANREINFORCE_H_

#include "rele/IRL/utils/hessian/HessianCalculator.h"

namespace ReLe
{
template<class ActionC, class StateC>
class HessianReinforce : public HessianCalculator<ActionC, StateC>
{
protected:
    USE_HESSIAN_CALCULATOR_MEMBERS(ActionC, StateC)

public:
    HessianReinforce(Features& phi,
                     Dataset<ActionC,StateC>& data,
                     DifferentiablePolicy<ActionC,StateC>& policy,
                     double gamma) : HessianCalculator<ActionC, StateC>(phi, data, policy, gamma)
    {

    }

    virtual ~HessianReinforce()
    {

    }

protected:
    virtual arma::cube computeHessianDiff() override
    {
        unsigned int dp = policy.getParametersSize();
        unsigned int dr = phi.rows();
        unsigned int episodeN = data.size();

        arma::cube Hdiff(dp, dp, phi.rows(), arma::fill::zeros);

        arma::mat Rew = data.computeEpisodeFeatureExpectation(phi, gamma);

        for(unsigned int ep = 0; ep < episodeN; ep++)
        {
            arma::mat G = this->computeG(data[ep]);

            for(unsigned int f = 0; f < phi.rows(); f++)
                Hdiff.slice(f) += G*Rew(f, ep);

        }

        Hdiff /= episodeN;

        return Hdiff;
    }



};

template<class ActionC, class StateC>
class HessianReinforceBase : public HessianCalculator<ActionC, StateC>
{
protected:
    USE_HESSIAN_CALCULATOR_MEMBERS(ActionC, StateC)

public:
    HessianReinforceBase(Features& phi,
                         Dataset<ActionC,StateC>& data,
                         DifferentiablePolicy<ActionC,StateC>& policy,
                         double gamma) : HessianCalculator<ActionC, StateC>(phi, data, policy, gamma)
    {

    }

    virtual ~HessianReinforceBase()
    {

    }

protected:
    virtual arma::cube computeHessianDiff() override
    {
        unsigned int dp = policy.getParametersSize();
        unsigned int dr = phi.rows();
        unsigned int episodeN = data.size();

        arma::cube Hdiff(dp, dp, dr, arma::fill::zeros);
        arma::mat Rew = data.computeEpisodeFeatureExpectation(phi, gamma);

        arma::cube baseline_num(dp, dp, dr, arma::fill::zeros);
        arma::mat baseline_den(dp, dp, arma::fill::zeros);
        arma::cube G_ep(dp, dp, episodeN);

        for(unsigned int ep = 0; ep < episodeN; ep++)
        {
            // compute hessian essential
            arma::mat G = this->computeG(data[ep]);
            arma::mat G2 = G % G;

            // store hessian essentials
            G_ep.slice(ep) = G;
            baseline_den += G2;

            for(unsigned int r = 0; r < dr; r++)
                baseline_num.slice(r) += Rew(r, ep)*G2;
        }

        // compute the hessian
        arma::cube baseline = baseline_num.each_slice() / baseline_den;
        baseline(arma::find_nonfinite(baseline)).zeros();

        for (int ep = 0; ep < episodeN; ep++)
        {
            for(int r = 0; r < phi.rows(); r++)
                Hdiff.slice(r) += (Rew(r, ep) - baseline.slice(r)) % G_ep.slice(ep);
        }

        // compute mean values
        Hdiff /= episodeN;

        return Hdiff;
    }
};

}



#endif /* INCLUDE_RELE_IRL_UTILS_HESSIAN_HESSIANREINFORCE_H_ */
