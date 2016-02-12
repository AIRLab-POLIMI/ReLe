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

#ifndef INCLUDE_RELE_IRL_UTILS_GRADIENT_ENACGRADIENTCALCULATOR_H_
#define INCLUDE_RELE_IRL_UTILS_GRADIENT_ENACGRADIENTCALCULATOR_H_

#include "rele/IRL/utils/gradient/GradientCalculator.h"

namespace ReLe
{


template<class ActionC, class StateC>
class ENACGradientCalculator : public GradientCalculator<ActionC, StateC>
{
    USE_GRADIENT_CALCULATOR_MEMBERS(ActionC, StateC)

public:
    ENACGradientCalculator(Features& phi,
                           Dataset<ActionC,StateC>& data,
                           DifferentiablePolicy<ActionC,StateC>& policy,
                           double gamma):
        GradientCalculator<ActionC, StateC>(phi, data, policy,gamma)
    {

    }

    virtual ~ENACGradientCalculator()
    {

    }

protected:
    virtual arma::mat computeGradientDiff() override
    {
        int dp  = policy.getParametersSize();
        arma::vec localg;
        double Rew;
        arma::vec g(dp+1, arma::fill::zeros), phiP(dp+1);
        arma::mat fisher(dp+1,dp+1, arma::fill::zeros);

        int nbEpisodes = data.size();
        for (int i = 0; i < nbEpisodes; ++i)
        {
            //core setup
            int nbSteps = data[i].size();

            double df = 1.0;
            Rew = 0.0;
            phiP.zeros();
            phiP(dp) = 1.0;

            //iterate the episode
            for (int t = 0; t < nbSteps; ++t)
            {
                Transition<ActionC, StateC>& tr = data[i][t];

                localg = policy.difflog(tr.x, tr.u);
                double creward = rewardf(vectorize(tr.x, tr.u, tr.xn));
                Rew += df * creward;

                //Construct basis functions
                for (unsigned int i = 0; i < dp; ++i)
                    phiP[i] += df * localg[i];

                df *= gamma;

                if (tr.xn.isAbsorbing())
                {
                    assert(nbSteps == t+1);
                    break;
                }
            }

            fisher += phiP * phiP.t();
            g += Rew * phiP;

        }


        arma::vec nat_grad;
        int rnk = arma::rank(fisher);
        if (rnk == fisher.n_rows)
        {
            nat_grad = arma::solve(fisher, g);
        }
        else
        {
            std::cerr << "WARNING: Fisher Matrix is lower rank (rank = " << rnk << ")!!! Should be " << fisher.n_rows << std::endl;

            arma::mat H = arma::pinv(fisher);
            nat_grad = H * g;
        }

        return nat_grad.rows(0,dp-1);
    }
};

template<class ActionC, class StateC>
class ENACBaseGradientCalculator : public GradientCalculator<ActionC, StateC>
{
    USE_GRADIENT_CALCULATOR_MEMBERS(ActionC, StateC)
public:
    ENACBaseGradientCalculator(Features& phi,
                               Dataset<ActionC,StateC>& data,
                               DifferentiablePolicy<ActionC,StateC>& policy,
                               double gamma):
        GradientCalculator<ActionC, StateC>(phi, data, policy,gamma)
    {

    }

    virtual ~ENACBaseGradientCalculator()
    {

    }

protected:
    virtual arma::mat computeGradientDiff() override
    {
        int dp  = policy.getParametersSize();
        arma::vec localg;
        double Rew;
        arma::vec g(dp+1, arma::fill::zeros), eligibility(dp+1, arma::fill::zeros), phiP(dp+1);
        arma::mat fisher(dp+1,dp+1, arma::fill::zeros);
        double Jpol = 0.0;

        int nbEpisodes = data.size();
        for (int i = 0; i < nbEpisodes; ++i)
        {
            //core setup
            int nbSteps = data[i].size();

            double df = 1.0;
            Rew = 0.0;
            phiP.zeros();
            phiP(dp) = 1.0;

            //iterate the episode
            for (int t = 0; t < nbSteps; ++t)
            {
                Transition<ActionC, StateC>& tr = data[i][t];

                localg = policy.difflog(tr.x, tr.u);
                double creward = rewardf(tr.x, tr.u, tr.xn);
                Rew += df * creward;

                //Construct basis functions
                for (unsigned int i = 0; i < dp; ++i)
                    phiP[i] += df * localg[i];

                df *= gamma;

                if (tr.xn.isAbsorbing())
                {
                    assert(nbSteps == t+1);
                    break;
                }
            }

            Jpol += Rew;
            fisher += phiP * phiP.t();
            g += Rew * phiP;
            eligibility += phiP;

        }

        // compute mean value
        fisher /= nbEpisodes;
        g /= nbEpisodes;
        eligibility /= nbEpisodes;
        Jpol /= nbEpisodes;

        arma::vec nat_grad;
        int rnk = arma::rank(fisher);

        if (rnk == fisher.n_rows)
        {
            arma::mat tmp = arma::solve(nbEpisodes * fisher - eligibility * eligibility.t(), eligibility);
            arma::mat Q = (1 + eligibility.t() * tmp) / nbEpisodes;
            arma::mat b = Q * (Jpol - eligibility.t() * arma::solve(fisher, g));
            arma::vec grad = g - eligibility * b;
            nat_grad = arma::solve(fisher, grad);
        }
        else
        {
            std::cerr << "WARNING: Fisher Matrix is lower rank (rank = " << rnk << ")!!! Should be " << fisher.n_rows << std::endl;

            arma::mat H = arma::pinv(fisher);
            arma::mat b = (1 + eligibility.t() * arma::pinv(nbEpisodes * fisher - eligibility * eligibility.t()) * eligibility)
                          * (Jpol - eligibility.t() * H * g)/ nbEpisodes;
            arma::vec grad = g - eligibility * b;
            nat_grad = H * (grad);
        }

        return nat_grad.rows(0,dp-1);
    }
};


}



#endif /* INCLUDE_RELE_IRL_UTILS_GRADIENT_ENACGRADIENTCALCULATOR_H_ */
