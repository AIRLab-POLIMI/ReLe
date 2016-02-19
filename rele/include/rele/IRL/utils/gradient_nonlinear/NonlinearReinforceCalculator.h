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

#ifndef INCLUDE_RELE_IRL_UTILS_GRADIENT_NONLINEAR_NONLINEARREINFORCECALCULATOR_H_
#define INCLUDE_RELE_IRL_UTILS_GRADIENT_NONLINEAR_NONLINEARREINFORCECALCULATOR_H_

#include "rele/IRL/utils/gradient_nonlinear/NonlinearGradientCalculator.h"

namespace ReLe
{

template<class ActionC, class StateC>
class NonlinearReinforceCalculator : public NonlinearGradientCalculator<ActionC,StateC>
{

protected:
    USE_NONLINEAR_GRADIENT_CALCULATOR_MEMBERS(ActionC, StateC)

public:
    NonlinearReinforceCalculator(ParametricRegressor& rewardFunc,
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

        unsigned int totStep = 0;
        unsigned int episodeN = data.size();

        //iterate episodes
        for (int i = 0; i < episodeN; ++i)
        {
            double Rew;
            arma::rowvec dRew(dr);
            arma::vec sumGradLog(dp);
            this->computeEpisodeStatistics(data[i], Rew, dRew, sumGradLog);

            gradient += sumGradLog * Rew;
            dGradient += sumGradLog * dRew;

            totStep += data[i].size();
        }

        // compute mean values
        this->normalizeGradient(totStep, episodeN);

    }

    virtual ~NonlinearReinforceCalculator()
    {

    }
};

template<class ActionC, class StateC>
class NonlinearReinforceBaseCalculator : public NonlinearGradientCalculator<ActionC,StateC>
{

protected:
    USE_NONLINEAR_GRADIENT_CALCULATOR_MEMBERS(ActionC, StateC)

public:
    NonlinearReinforceBaseCalculator(ParametricRegressor& rewardFunc,
                                     Dataset<ActionC,StateC>& data,
                                     DifferentiablePolicy<ActionC,StateC>& policy,
                                     double gamma) : NonlinearGradientCalculator<ActionC,StateC>(rewardFunc, data, policy, gamma)
    {

    }

    virtual void compute() override
    {
        int dp = policy.getParametersSize();
        int dr = rewardFunc.getParametersSize();
        int episodeN = data.size();

        // Reset computed results
        gradient.zeros(dp);
        dGradient.zeros(dp, dr);

        // gradient basics
        arma::vec Rew_ep(episodeN);
        arma::mat dRew_ep(episodeN, dr);
        arma::mat sumGradLog_ep(dp, episodeN);

        // baselines
        arma::vec baseline_den(dp, arma::fill::zeros);
        arma::vec baseline_num_Rew(dp, arma::fill::zeros);
        arma::mat baseline_num_dRew(dp, dr, arma::fill::zeros);


        int totStep = 0;
        for (int i = 0; i < episodeN; ++i)
        {
            // compute basic elements
            double Rew;
            arma::rowvec dRew(dr);
            arma::vec sumGradLog(dp);
            this->computeEpisodeStatistics(data[i], Rew, dRew, sumGradLog);

            // store them
            Rew_ep(i) = Rew;
            dRew_ep.row(i) = dRew;
            sumGradLog_ep.col(i) = sumGradLog;

            // compute baseline num and den
            arma::vec sumGradLog2 = sumGradLog % sumGradLog;
            baseline_den += sumGradLog2;
            baseline_num_Rew += sumGradLog2 * Rew;
            baseline_num_dRew += sumGradLog2 * dRew;

            // update total step count
            totStep += data[i].size();
        }

        // compute the gradients
        arma::vec baseline = baseline_num_Rew / baseline_den;
        baseline(arma::find_nonfinite(baseline)).zeros();

        arma::mat baseline_d = baseline_num_dRew.each_col() / baseline_den;
        baseline_d(arma::find_nonfinite(baseline_d)).zeros();

        for (int ep = 0; ep < episodeN; ep++)
        {
            gradient += (Rew_ep(ep) - baseline) % sumGradLog_ep.col(ep);

            for(unsigned int r = 0; r < dr; r++)
                dGradient.col(r) += (dRew_ep(ep, r) - baseline_d.col(r)) % sumGradLog_ep.col(ep);
        }

        // compute mean values
        this->normalizeGradient(totStep, episodeN);

    }

    virtual ~NonlinearReinforceBaseCalculator()
    {

    }
};

}

#endif /* INCLUDE_RELE_IRL_UTILS_GRADIENT_NONLINEAR_NONLINEARREINFORCECALCULATOR_H_ */
