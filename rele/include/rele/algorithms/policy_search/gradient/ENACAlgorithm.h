/*
 * rele,
 *
 *
 * Copyright (C) 2015 Davide Tateo & Matteo Pirotta
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

#ifndef ENACALGORITHM_H_
#define ENACALGORITHM_H_

#include "rele/algorithms/policy_search/gradient/PolicyGradientAlgorithm.h"

namespace ReLe
{

///////////////////////////////////////////////////////////////////////////////////////
/// eNAC GRADIENT ALGORITHM
///////////////////////////////////////////////////////////////////////////////////////
/**
 * Policy Gradient Methods for Robotics
 * Jan Peters, Stefan Schaal
 * IROS 2006
 * http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/IROS2006-Peters_%5b0%5d.pdf
 */

template<class ActionC, class StateC>
class eNACAlgorithm: public AbstractPolicyGradientAlgorithm<ActionC, StateC>
{
    USE_PGA_MEMBERS

public:
    eNACAlgorithm(DifferentiablePolicy<ActionC, StateC>& policy,
                  unsigned int nbEpisodes, GradientStep& stepL,
                  bool baseline = true, int reward_obj = 0) :
        AbstractPolicyGradientAlgorithm<ActionC, StateC>(policy, nbEpisodes, stepL, baseline, reward_obj)
    {
        Jpol = 0;
    }

    eNACAlgorithm(DifferentiablePolicy<ActionC, StateC>& policy,
                  unsigned int nbEpisodes, GradientStep& stepL,
                  RewardTransformation& reward_tr,
                  bool baseline = true) :
        AbstractPolicyGradientAlgorithm<ActionC, StateC>(policy, nbEpisodes, stepL, reward_tr, baseline)
    {
        Jpol = 0;
    }

    virtual ~eNACAlgorithm()
    {
    }

    // Agent interface
protected:
    virtual void init() override
    {
        unsigned int dp = policy.getParametersSize();
        AbstractPolicyGradientAlgorithm<ActionC, StateC>::init();
        history_sumdlogpi.assign(nbEpisodesToEvalPolicy,arma::vec(dp));

        fisher.zeros(dp+1,dp+1);
        g.zeros(dp+1);
        eligibility.zeros(dp+1);
        psi.zeros(dp+1);

        Jpol = 0.0;
    }

    virtual void initializeVariables() override
    {
        psi.zeros();
        psi(policy.getParametersSize()) = 1.0;
    }

    virtual void updateStep(const Reward& reward) override
    {
        // get number of parameters
        int dp = policy.getParametersSize();

        // Compute the derivative of the logarithm of the policy and
        // Evaluate it in (s_t, a_t)
        arma::vec grad = policy.difflog(currentState, currentAction);

        //Construct basis functions
        psi.rows(0, dp-1) += grad;
    }

    virtual void updateAtEpisodeEnd() override
    {
        Jpol += Jep;
        fisher += psi * psi.t();
        g += Jep * psi;
        eligibility += psi;
    }

    virtual void updatePolicy() override
    {
        // get number of parameters
        int dp = policy.getParametersSize();

        // compute mean value
        if (task.gamma == 1.0)
        {
            fisher /= totstep;
            g /= totstep;
            eligibility /= totstep;
            Jpol /= totstep;
        }
        else
        {
            fisher /= nbEpisodesToEvalPolicy;
            g /= nbEpisodesToEvalPolicy;
            eligibility /= nbEpisodesToEvalPolicy;
            Jpol /= nbEpisodesToEvalPolicy;
        }
        int nbParams = policy.getParametersSize();

        //--- Compute learning step

        arma::vec gradient;
        arma::vec nat_grad;
        int rnk = arma::rank(fisher);
        if (rnk == fisher.n_rows)
        {
            if (useBaseline == true)
            {
                arma::mat tmp = arma::solve(nbEpisodesToEvalPolicy * fisher - eligibility * eligibility.t(), eligibility);
                arma::mat Q = (1 + eligibility.t() * tmp) / nbEpisodesToEvalPolicy;
                arma::mat b = Q * (Jpol - eligibility.t() * arma::solve(fisher, g));
                gradient = g - eligibility * b;
                nat_grad = arma::solve(fisher, gradient);
            }
            else
            {
                gradient = g;
                nat_grad = arma::solve(fisher, gradient);
            }
        }
        else
        {
            std::cerr << "WARNING: Fisher Matrix is lower rank (rank = " << rnk << ")!!! Should be " << fisher.n_rows << std::endl;

            arma::mat H = arma::pinv(fisher);

            if (useBaseline == true)
            {
                arma::mat b = (1 + eligibility.t() * arma::pinv(nbEpisodesToEvalPolicy * fisher - eligibility * eligibility.t()) * eligibility)
                              * (Jpol - eligibility.t() * H * g)/ nbEpisodesToEvalPolicy;
                gradient = g - eligibility * b;
                nat_grad = H * (gradient);
            }
            else
            {
                gradient = g;
                nat_grad = H * gradient;
            }
        }

        //--- save actual policy performance
        currentItStats->history_J = history_J;
        currentItStats->history_gradients = history_sumdlogpi;
        currentItStats->estimated_gradient = nat_grad.rows(0,dp-1);

        arma::vec delta = stepLength(gradient, nat_grad);
        arma::vec newvalues = policy.getParameters() + delta.rows(0,dp-1);
        policy.setParameters(newvalues);

        eligibility.zeros();
        g.zeros();

        fisher.zeros();
        Jpol = 0.0;
    }

protected:
    std::vector<arma::vec> history_sumdlogpi;
    arma::vec g, eligibility, psi;

    arma::mat fisher;
    double Jpol;
};

}// end namespace ReLe

#endif //ENACALGORITHM_H_
