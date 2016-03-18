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

#ifndef INCLUDE_RELE_IRL_UTILS_GRADIENT_REINFORCEGRADIENTCALCULATOR_H_
#define INCLUDE_RELE_IRL_UTILS_GRADIENT_REINFORCEGRADIENTCALCULATOR_H_

#include "rele/IRL/utils/gradient/step_based/StepBasedGradientCalculator.h"

namespace ReLe
{

template<class ActionC, class StateC>
class ReinforceGradientCalculator : public StepBasedGradientCalculator<ActionC, StateC>
{
protected:
    USING_STEP_BASED_CALCULATORS_MEMBERS(ActionC, StateC)

public:
    ReinforceGradientCalculator(Features& phi,
                                Dataset<ActionC,StateC>& data,
                                DifferentiablePolicy<ActionC,StateC>& policy,
                                double gamma):
        StepBasedGradientCalculator<ActionC, StateC>(phi, data, policy, gamma)
    {

    }

    virtual ~ReinforceGradientCalculator()
    {

    }

protected:
    virtual arma::mat computeGradientDiff() override
    {
        arma::mat Rew = data.computeEpisodeFeatureExpectation(phi, gamma);
        arma::mat gradient(policy.getParametersSize(), phi.rows(), arma::fill::zeros);

        int nbEpisodes = data.size();
        for (int i = 0; i < nbEpisodes; ++i)
        {
            arma::vec sumGradLog = this->computeSumGradLog(data[i]);
            gradient += sumGradLog * Rew.col(i).t();
        }

        // compute mean values
        gradient /= nbEpisodes;

        return gradient;
    }

};

template<class ActionC, class StateC>
class ReinforceBaseGradientCalculator : public ReinforceGradientCalculator<ActionC, StateC>
{
protected:
    USING_STEP_BASED_CALCULATORS_MEMBERS(ActionC, StateC)

public:
    ReinforceBaseGradientCalculator(Features& phi,
                                    Dataset<ActionC,StateC>& data,
                                    DifferentiablePolicy<ActionC,StateC>& policy,
                                    double gamma):
        ReinforceGradientCalculator<ActionC, StateC>(phi, data, policy, gamma)
    {

    }

    virtual ~ReinforceBaseGradientCalculator()
    {

    }

protected:
    virtual arma::mat computeGradientDiff() override
    {
        int dp = policy.getParametersSize();
        int nbEpisodes = data.size();

        arma::mat Rew = data.computeEpisodeFeatureExpectation(phi, gamma);
        arma::mat gradient(dp, phi.rows(), arma::fill::zeros);

        arma::mat baseline_num(dp, phi.rows(), arma::fill::zeros);
        arma::vec baseline_den(dp, arma::fill::zeros);
        arma::mat sumGradLog_Ep(dp,nbEpisodes);

        for (int i = 0; i < nbEpisodes; ++i)
        {
            arma::vec sumGradLog = this->computeSumGradLog(data[i]);
            sumGradLog_Ep.col(i) = sumGradLog;

            // compute the baselines
            baseline_num += (sumGradLog % sumGradLog) * Rew.col(i).t() ;
            baseline_den += sumGradLog % sumGradLog;
        }

        // compute the gradients
        arma::mat baseline = baseline_num.each_col() / baseline_den;
        baseline(arma::find_nonfinite(baseline)).zeros();

        for (int ep = 0; ep < nbEpisodes; ep++)
        {
            for(int r = 0; r < phi.rows(); r++)
                gradient.col(r) += (Rew(r, ep) - baseline.col(r)) % sumGradLog_Ep.col(ep);
        }


        // compute mean values
        gradient /= nbEpisodes;

        return gradient;
    }

};


}


#endif /* INCLUDE_RELE_IRL_UTILS_GRADIENT_REINFORCEGRADIENTCALCULATOR_H_ */
