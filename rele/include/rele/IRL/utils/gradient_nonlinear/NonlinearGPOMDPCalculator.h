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

#ifndef INCLUDE_RELE_IRL_UTILS_GRADIENT_NONLINEAR_NONLINEARGPOMDPCALCULATOR_H_
#define INCLUDE_RELE_IRL_UTILS_GRADIENT_NONLINEAR_NONLINEARGPOMDPCALCULATOR_H_

#include "rele/IRL/utils/gradient_nonlinear/NonlinearGradientCalculator.h"


namespace ReLe
{

template<class ActionC, class StateC>
class NonlinearGPOMDPCalculator : public NonlinearGradientCalculator<ActionC, StateC>
{
protected:
    USE_NONLINEAR_GRADIENT_CALCULATOR_MEMBERS(ActionC, StateC)
public:
    NonlinearGPOMDPCalculator(ParametricRegressor& rewardFunc,
                              Dataset<ActionC,StateC>& data,
                              DifferentiablePolicy<ActionC,StateC>& policy,
                              double gamma) : NonlinearGradientCalculator<ActionC,StateC>(rewardFunc, data, policy, gamma)
    {

    }

    virtual void compute() override
    {
        int dp = policy.getParametersSize();
        int dr = rewardFunc.getParametersSize();

        // Reset computed results
        gradient.zeros(dp);
        dGradient.zeros(dp, dr);

        int totStep = 0;
        int episodeN = data.size();
        for (int i = 0; i < episodeN; ++i)
        {
            //core setup
            int stepN = data[i].size();
            arma::vec sumGradLog(dp, arma::fill::zeros);

            //iterate the episode
            double df = 1.0;

            for (int t = 0; t < stepN; ++t)
            {
                Transition<ActionC, StateC>& tr = data[i][t];

                // compute the reward gradients
                double Rew = df * arma::as_scalar(rewardFunc(tr.x, tr.u, tr.xn));
                arma::rowvec dRew = df * rewardFunc.diff(tr.x, tr.u, tr.xn).t();
                sumGradLog += policy.difflog(tr.x, tr.u);

                gradient += sumGradLog * Rew;
                dGradient += sumGradLog * dRew;

                df *= gamma;
            }

            totStep += stepN;
        }

        // compute mean values
        this->normalizeGradient(totStep, episodeN);
    }

    virtual ~NonlinearGPOMDPCalculator()
    {

    }
};

template<class ActionC, class StateC>
class NonlinearGPOMDPBaseCalculator : public NonlinearGradientCalculator<ActionC,StateC>
{
protected:
    USE_NONLINEAR_GRADIENT_CALCULATOR_MEMBERS(ActionC, StateC)

public:
    NonlinearGPOMDPBaseCalculator(ParametricRegressor& rewardFunc,
                                  Dataset<ActionC,StateC>& data,
                                  DifferentiablePolicy<ActionC,StateC>& policy,
                                  double gamma) : NonlinearGradientCalculator<ActionC,StateC>(rewardFunc, data, policy, gamma)
    {

    }

    virtual void compute() override
    {
        int dp = policy.getParametersSize();
        int dr = rewardFunc.getParametersSize();
        unsigned int maxSteps = data.getEpisodeMaxLength();

        int episodeN = data.size();

        // Reset computed results
        gradient.zeros(dp);
        dGradient.zeros(dp, dr);

        // gradient basics
        arma::mat Rew_epStep(episodeN, maxSteps);
        arma::cube dRew_epStep(episodeN, maxSteps, dr);
        arma::cube sumGradLog_epStep(episodeN, maxSteps, dp);
        arma::vec maxsteps_Ep(episodeN);

        // baseline
        arma::mat baseline_den(dp, maxSteps, arma::fill::zeros);
        arma::mat baseline_num_Rew(dp, maxSteps, arma::fill::zeros);
        arma::cube baseline_num_dRew(dp, dr, maxSteps, arma::fill::zeros);

        int totStep = 0;
        for (int ep = 0; ep < episodeN; ++ep)
        {
            //core setup
            int stepN = data[ep].size();

            arma::vec sumGradLog(dp, arma::fill::zeros);

            //iterate the episode
            double df = 1.0;

            for (int t = 0; t < stepN; ++t)
            {
                Transition<ActionC, StateC>& tr = data[ep][t];

                // compute the basic elements used to compute the gradients
                double Rew = df * arma::as_scalar(rewardFunc(tr.x, tr.u, tr.xn));
                arma::rowvec dRew = df * rewardFunc.diff(tr.x, tr.u, tr.xn).t();
                sumGradLog += policy.difflog(tr.x, tr.u);

                // store the basic elements used to compute the gradients
                Rew_epStep(ep, t) = Rew;
                dRew_epStep.tube(ep, t)= dRew;
                sumGradLog_epStep.tube(ep, t) = sumGradLog;

                // compute the baselines
                arma::vec sumGradLog2 = sumGradLog % sumGradLog;
                baseline_num_Rew.col(t) += sumGradLog2 * Rew;
                baseline_num_dRew.slice(t) += sumGradLog2 * dRew;
                baseline_den.col(t) += sumGradLog2;

                df *= gamma;
            }

            // store the actual length of the current episode (<= maxsteps)
            maxsteps_Ep(ep) = stepN;

            totStep += stepN;

        }

        // compute the gradients
        for (int ep = 0; ep < episodeN; ++ep)
        {
            for (int t = 0; t < maxsteps_Ep(ep); ++t)
            {
                // compute the gradients
                arma::vec baseline_t = baseline_num_Rew.col(t) / baseline_den.col(t);
                baseline_t(arma::find_nonfinite(baseline_t)).zeros();

                arma::mat baseline_d_t = baseline_num_dRew.slice(t).each_col() / baseline_den.col(t);
                baseline_d_t(arma::find_nonfinite(baseline_d_t)).zeros();

                arma::vec sumGradLog_ep_t = sumGradLog_epStep.tube(ep,t);
                arma::vec dRew_ep_t = dRew_epStep.tube(ep, t);

                gradient += sumGradLog_ep_t % (Rew_epStep(ep, t) - baseline_t);
                for (unsigned int r = 0; r < dr; r++)
                    dGradient.col(r) += sumGradLog_ep_t % (dRew_ep_t(r) - baseline_d_t.col(r));

            }
        }

        // compute mean values
        this->normalizeGradient(totStep, episodeN);

    }

    virtual ~NonlinearGPOMDPBaseCalculator()
    {

    }
};


}


#endif /* INCLUDE_RELE_IRL_UTILS_GRADIENT_NONLINEAR_NONLINEARGPOMDPCALCULATOR_H_ */
