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

#ifndef INCLUDE_RELE_IRL_UTILS_NONLINEARGRADIENTCALCULATOR_H_
#define INCLUDE_RELE_IRL_UTILS_NONLINEARGRADIENTCALCULATOR_H_

#include "rele/policy/Policy.h"
#include "rele/core/Transition.h"
#include "rele/approximators/Features.h"
#include "rele/approximators/Regressors.h"
#include <cassert>

namespace ReLe
{

template<class ActionC, class StateC>
class NonlinearGradientCalculator
{
public:
    NonlinearGradientCalculator(ParametricRegressor& rewardFunc,
                                Dataset<ActionC,StateC>& data,
                                DifferentiablePolicy<ActionC,StateC>& policy,
                                double gamma) : rewardFunc(rewardFunc), data(data), policy(policy), gamma(gamma)
    {

    }

    virtual void compute() = 0;

    inline arma::vec getGradient()
    {
        return gradient;
    }

    inline arma::mat getGradientDiff()
    {
        return dGradient;
    }

    virtual ~NonlinearGradientCalculator()
    {

    }

protected:
    void normalizeGradient(unsigned int totStep, unsigned int episodeN)
    {
        if (gamma == 1.0)
        {
            gradient /= totStep;
            dGradient /= totStep;
        }
        else
        {
            gradient /= episodeN;
            dGradient /= episodeN;
        }
    }

    void computeEpisodeStatistics(Episode<ActionC,StateC>& episode, double& Rew,  arma::rowvec& dRew, arma::vec& sumGradLog)
    {
        //reste data
        Rew = 0;
        dRew.zeros();
        sumGradLog.zeros();


        double df = 1.0;

        //iterate the episode
        for (int t = 0; t < episode.size(); ++t)
        {
            Transition<ActionC, StateC>& tr = episode[t];
            sumGradLog += policy.difflog(tr.x, tr.u);

            Rew += df * arma::as_scalar(rewardFunc(tr.x, tr.u, tr.xn));
            dRew += df * rewardFunc.diff(tr.x, tr.u, tr.xn).t();

            df *= gamma;
        }
    }


protected:
    ParametricRegressor& rewardFunc;
    Dataset<ActionC,StateC>& data;
    DifferentiablePolicy<ActionC,StateC>& policy;
    double gamma;

    arma::vec gradient;
    arma::mat dGradient;

};

}

#define USE_NONLINEAR_GRADIENT_CALCULATOR_MEMBERS(ActionC, StateC) \
			typedef NonlinearGradientCalculator<ActionC,StateC> Base; \
			using Base::rewardFunc; \
			using Base::data; \
			using Base::policy; \
			using Base::gamma; \
			using Base::gradient; \
			using Base::dGradient;


#endif /* INCLUDE_RELE_IRL_UTILS_NONLINEARGRADIENTCALCULATOR_H_ */
