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

#include "PolicyGradientAlgorithm.h"

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

#define AUGMENTED

template<class ActionC, class StateC>
class eNACAlgorithm: public AbstractPolicyGradientAlgorithm<ActionC, StateC>
{
    USE_PGA_MEMBERS

public:
    eNACAlgorithm(DifferentiablePolicy<ActionC, StateC>& policy,
                  unsigned int nbEpisodes, StepRule& stepL,
                  bool baseline = true, int reward_obj = 0) :
        AbstractPolicyGradientAlgorithm<ActionC, StateC>(policy, nbEpisodes, stepL, baseline, reward_obj)
    {
        Jpol = 0;
    }

    eNACAlgorithm(DifferentiablePolicy<ActionC, StateC>& policy,
                  unsigned int nbEpisodes, StepRule& stepL,
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
    virtual void init()
    {
        unsigned int dp = policy.getParametersSize();
        AbstractPolicyGradientAlgorithm<ActionC, StateC>::init();
        history_sumdlogpi.assign(nbEpisodesToEvalPolicy,arma::vec(dp));
#ifdef AUGMENTED
        fisher.zeros(dp+1,dp+1);
        g.zeros(dp+1);
        eligibility.zeros(dp+1);
        phi.zeros(dp+1);
#else
        fisher.zeros(dp,dp);
        g.zeros(dp);
        eligibility.zeros(dp);
        phi.zeros(dp);
#endif
        Jpol = 0.0;
    }

    virtual void initializeVariables()
    {
        phi.zeros();
#ifdef AUGMENTED
        phi(policy.getParametersSize()) = 1.0;
#endif
    }

    virtual void updateStep(const Reward& reward)
    {
        // get number of parameters
        int dp = policy.getParametersSize();

        // Compute the derivative of the logarithm of the policy and
        // Evaluate it in (s_t, a_t)
        arma::vec grad = policy.difflog(currentState, currentAction);

        //Construct basis functions
        for (unsigned int i = 0; i < dp; ++i)
            phi[i] += df * grad[i];
    }

    virtual void updateAtEpisodeEnd()
    {
        Jpol += Jep;
        fisher += phi * phi.t();
        g += Jep * phi;
        eligibility += phi;
    }

    virtual void updatePolicy()
    {
        // get number of parameters
        int dp = policy.getParametersSize();

        // compute mean value
        fisher /= nbEpisodesToEvalPolicy;
        g /= nbEpisodesToEvalPolicy;
        eligibility /= nbEpisodesToEvalPolicy;
        Jpol /= nbEpisodesToEvalPolicy;
        int nbParams = policy.getParametersSize();

        //--- Compute learning step

        arma::vec step_size;
        arma::vec nat_grad;
        int rnk = arma::rank(fisher);
        //        std::cout << rnk << " " << fisher << std::endl;
        if (rnk == fisher.n_rows)
        {
            arma::vec grad;
            if (useBaseline == true)
            {
                arma::mat tmp = arma::solve(nbEpisodesToEvalPolicy * fisher - eligibility * eligibility.t(), eligibility);
                arma::mat Q = (1 + eligibility.t() * tmp) / nbEpisodesToEvalPolicy;
                arma::mat b = Q * (Jpol - eligibility.t() * arma::solve(fisher, g));
                grad = g - eligibility * b;
                nat_grad = arma::solve(fisher, grad);
            }
            else
            {
                grad = g;
                nat_grad = arma::solve(fisher, grad);
            }

            step_size = stepLength.stepLength(grad, fisher);
        }
        else
        {
            std::cerr << "WARNING: Fisher Matrix is lower rank (rank = " << rnk << ")!!! Should be " << fisher.n_rows << std::endl;

            arma::mat H = arma::pinv(fisher);
            arma::vec grad;
            if (useBaseline == true)
            {
                arma::mat b = (1 + eligibility.t() * arma::pinv(nbEpisodesToEvalPolicy * fisher - eligibility * eligibility.t()) * eligibility)
                              * (Jpol - eligibility.t() * H * g)/ nbEpisodesToEvalPolicy;
                grad = g - eligibility * b;
                nat_grad = H * (grad);
            }
            else
            {
                grad = g;
                nat_grad = H * grad;
            }
            step_size = stepLength.stepLength(grad, fisher);
        }
        //---

        //--- save actual policy performance
        currentItStats->history_J = history_J;
        currentItStats->history_gradients = history_sumdlogpi;
        currentItStats->estimated_gradient = nat_grad.rows(0,dp-1);
        currentItStats->stepLength = step_size;
        //---

        //        std::cout << stepLength <<std::endl;
        //        std::cout << nat_grad.t();

        arma::vec newvalues = policy.getParameters() + nat_grad.rows(0,dp-1) * step_size;
        policy.setParameters(newvalues);
        //        std::cout << "new_params: "  << newvalues.t();

        for (int p = 0; p < nbParams; ++p)
        {
            eligibility[p] = 0.0;
            g[p] = 0.0;
        }
        fisher.zeros();
        Jpol = 0.0;
    }

protected:
    std::vector<arma::vec> history_sumdlogpi;
    arma::vec g, eligibility, phi;

    arma::mat fisher;
    double Jpol;
};

}// end namespace ReLe

#endif //ENACALGORITHM_H_
