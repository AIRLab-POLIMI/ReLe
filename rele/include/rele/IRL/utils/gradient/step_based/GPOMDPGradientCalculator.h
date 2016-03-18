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

#ifndef INCLUDE_RELE_IRL_UTILS_GRADIENT_GPOMDPGRADIENTCALCULATOR_H_
#define INCLUDE_RELE_IRL_UTILS_GRADIENT_GPOMDPGRADIENTCALCULATOR_H_

#include "rele/IRL/utils/gradient/step_based/StepBasedGradientCalculator.h"

namespace ReLe
{


template<class ActionC, class StateC>
class GPOMDPGradientCalculator : public StepBasedGradientCalculator<ActionC, StateC>
{

protected:
    USING_STEP_BASED_CALCULATORS_MEMBERS(ActionC, StateC)


public:
    GPOMDPGradientCalculator(Features& phi,
                             Dataset<ActionC,StateC>& data,
                             DifferentiablePolicy<ActionC,StateC>& policy,
                             double gamma):
        StepBasedGradientCalculator<ActionC, StateC>(phi, data, policy, gamma)
    {

    }

    virtual ~GPOMDPGradientCalculator()
    {

    }

protected:
    virtual arma::mat computeGradientDiff() override
    {
        unsigned int dp  = policy.getParametersSize();
        unsigned int dr = phi.rows();
        int nbEpisodes = data.size();

        arma::mat gradient(dp, dr, arma::fill::zeros);

        for (auto& episode : data)
        {
            //core setup
            arma::vec sumGradLog(dp, arma::fill::zeros);
            double df = 1.0;

            //iterate the episode
            for (auto& tr : episode)
            {
                sumGradLog += policy.difflog(tr.x, tr.u);
                arma::vec creward = phi(tr.x, tr.u, tr.xn);

                // compute the gradients
                gradient += df * sumGradLog * creward.t();

                df *= gamma;
            }

        }

        // compute mean values
        gradient /= nbEpisodes;

        return gradient;
    }
};

template<class ActionC, class StateC>
class GPOMDPBaseGradientCalculator : public StepBasedGradientCalculator<ActionC, StateC>
{
protected:
    USING_STEP_BASED_CALCULATORS_MEMBERS(ActionC, StateC)

public:
    GPOMDPBaseGradientCalculator(Features& phi,
                                 Dataset<ActionC,StateC>& data,
                                 DifferentiablePolicy<ActionC,StateC>& policy,
                                 double gamma):
        StepBasedGradientCalculator<ActionC, StateC>(phi, data, policy, gamma)
    {

    }

    virtual ~GPOMDPBaseGradientCalculator()
    {

    }

protected:
    virtual arma::mat computeGradientDiff() override
    {
        int dp  = policy.getParametersSize();
        unsigned int dr = phi.rows();
        int nbEpisodes = data.size();

        int maxSteps = data.getEpisodeMaxLength();

        arma::mat gradient(dp, dr, arma::fill::zeros);

        arma::cube baseline_num(dp, dr, maxSteps, arma::fill::zeros);
        arma::mat baseline_den(dp, maxSteps, arma::fill::zeros);

        arma::cube reward_EpStep(nbEpisodes, maxSteps, dr);
        arma::cube sumGradLog_EpStep(nbEpisodes, maxSteps, dp);
        arma::vec  maxsteps_Ep(nbEpisodes);

        for (int ep = 0; ep < nbEpisodes; ++ep)
        {
            //core setup
            int nbSteps = data[ep].size();

            double df = 1.0;
            arma::vec sumGradLog(dp, arma::fill::zeros);

            //iterate the episode
            for (int t = 0; t < nbSteps; ++t)
            {
                Transition<ActionC, StateC>& tr = data[ep][t];
                sumGradLog += policy.difflog(tr.x, tr.u);

                // Store the basic elements used to compute the gradients
                arma::vec creward = phi(tr.x, tr.u, tr.xn);
                reward_EpStep.tube(ep,t) = df * creward;
                sumGradLog_EpStep.tube(ep,t) = sumGradLog;

                // Compute baseline elements
                baseline_num.slice(t) += df * (sumGradLog % sumGradLog) * creward.t();
                baseline_den.col(t) += sumGradLog % sumGradLog;

                df *= gamma;
            }

            // store the actual length of the current episode (<= maxsteps)
            maxsteps_Ep(ep) = nbSteps;
        }

        // compute the gradients
        for (int ep = 0; ep < nbEpisodes; ++ep)
        {
            for (int t = 0; t < maxsteps_Ep(ep); ++t)
            {
                //Compute baseline
                arma::mat baseline = baseline_num.slice(t).each_col() / baseline_den.col(t);
                baseline(arma::find_nonfinite(baseline)).zeros();

                //Get sum of gradients of logarithms of the current step
                arma::vec sumGradLog_ep_t = sumGradLog_EpStep.tube(ep,t);

                //Compute gradient
                for(int r = 0; r < dr; r++)
                    gradient.col(r) += (reward_EpStep(ep,t,r) - baseline.col(r)) % sumGradLog_ep_t;
            }
        }

        // compute mean values
        gradient /= nbEpisodes;

        return gradient;
    }

};


}



#endif /* INCLUDE_RELE_IRL_UTILS_GRADIENT_GPOMDPGRADIENTCALCULATOR_H_ */
