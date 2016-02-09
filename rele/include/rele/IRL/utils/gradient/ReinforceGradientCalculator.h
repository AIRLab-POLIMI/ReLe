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

#include "rele/IRL/utils/gradient/GradientCalculator.h"

namespace ReLe
{


template<class ActionC, class StateC>
class ReinforceGradientCalculator : public GradientCalculator<ActionC, StateC>
{
    USE_GRADIENT_CALCULATOR_MEMBERS(ActionC, StateC)
public:
    ReinforceGradientCalculator(BasisFunctions& basis,
                                Dataset<ActionC,StateC>& data,
                                DifferentiablePolicy<ActionC,StateC>& policy,
                                double gamma):
        GradientCalculator<ActionC, StateC>(basis, data, policy,gamma)
    {

    }

    virtual ~ReinforceGradientCalculator()
    {

    }

protected:
    virtual arma::vec computeGradientFeature(BasisFunction& rewardf) override
    {
        int dp  = policy.getParametersSize();
        arma::vec sumGradLog(dp), localg;
        arma::vec gradient_J(dp, arma::fill::zeros);
        double Rew;

        int nbEpisodes = data.size();
        for (int i = 0; i < nbEpisodes; ++i)
        {
            //core setup
            int nbSteps = data[i].size();

            sumGradLog.zeros();
            double df = 1.0;
            Rew = 0.0;

            //iterate the episode
            for (int t = 0; t < nbSteps; ++t)
            {
                Transition<ActionC, StateC>& tr = data[i][t];
                localg = policy.difflog(tr.x, tr.u);
                sumGradLog += localg;
                Rew += df * rewardf(vectorize(tr.x, tr.u, tr.xn));

                df *= gamma;
            }

            for (int p = 0; p < dp; ++p)
            {
                gradient_J[p] += Rew * sumGradLog(p);
            }

        }

        // compute mean values
        gradient_J /= nbEpisodes;

        return gradient_J;
    }

};

template<class ActionC, class StateC>
class ReinforceBaseGradientCalculator : public GradientCalculator<ActionC, StateC>
{
    USE_GRADIENT_CALCULATOR_MEMBERS(ActionC, StateC)
public:
    ReinforceBaseGradientCalculator(BasisFunctions& basis,
                                    Dataset<ActionC,StateC>& data,
                                    DifferentiablePolicy<ActionC,StateC>& policy,
                                    double gamma):
        GradientCalculator<ActionC, StateC>(basis, data, policy,gamma)
    {

    }

    virtual ~ReinforceBaseGradientCalculator()
    {

    }

protected:
    virtual arma::vec computeGradientFeature(BasisFunction& rewardf) override
    {
        int dp  = policy.getParametersSize();
        int nbEpisodes = data.size();

        arma::vec sumGradLog(dp), localg;
        arma::vec gradient_J(dp, arma::fill::zeros);
        double Rew;

        arma::vec baseline_J_num(dp, arma::fill::zeros);
        arma::vec baseline_den(dp, arma::fill::zeros);
        arma::vec return_J_ObjEp(nbEpisodes);
        arma::mat sumGradLog_CompEp(dp,nbEpisodes);

        for (int i = 0; i < nbEpisodes; ++i)
        {
            //core setup
            int nbSteps = data[i].size();

            sumGradLog.zeros();
            double df = 1.0;
            Rew = 0.0;

            //iterate the episode
            for (int t = 0; t < nbSteps; ++t)
            {
                Transition<ActionC, StateC>& tr = data[i][t];
                //            std::cout << tr.x << " " << tr.u << " " << tr.xn << " " << tr.r[0] << std::endl;

                // *** REINFORCE CORE *** //
                localg = policy.difflog(tr.x, tr.u);
                for (int p = 0; p < dp; ++p)
                    assert(!isinf(localg(p)));
                sumGradLog += localg;
                Rew += df * rewardf(vectorize(tr.x, tr.u, tr.xn));
                // ********************** //

                df *= gamma;

                if (tr.xn.isAbsorbing())
                {
                    assert(nbSteps == t+1);
                    break;
                }
            }

            // store the basic elements used to compute the gradients

            return_J_ObjEp(i) = Rew;

            for (int p = 0; p < dp; ++p)
            {
                sumGradLog_CompEp(p,i) = sumGradLog(p);
            }

            // compute the baselines
            for (int p = 0; p < dp; ++p)
            {
                baseline_J_num(p) += Rew * sumGradLog(p) * sumGradLog(p);
                baseline_den(p) += sumGradLog(p) * sumGradLog(p);
                assert(!isinf(baseline_J_num(p)));
            }

        }

        // compute the gradients
        for (int p = 0; p < dp; ++p)
        {

            double baseline_J = 0;
            if (baseline_den(p) != 0)
            {
                baseline_J = baseline_J_num(p) / baseline_den(p);
            }

            for (int ep = 0; ep < nbEpisodes; ++ep)
            {
                double a =return_J_ObjEp(ep);
                double b = sumGradLog_CompEp(p,ep);
                assert(!isnan(a));
                assert(!isnan(b));
                assert(!isnan(baseline_J));
                gradient_J[p] += (return_J_ObjEp(ep) - baseline_J) * sumGradLog_CompEp(p,ep);
            }
        }

        // compute mean values
        gradient_J /= nbEpisodes;

        return gradient_J;


    }
};


}


#endif /* INCLUDE_RELE_IRL_UTILS_GRADIENT_REINFORCEGRADIENTCALCULATOR_H_ */
