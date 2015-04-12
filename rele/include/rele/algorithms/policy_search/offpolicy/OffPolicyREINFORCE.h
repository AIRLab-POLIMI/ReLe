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

#ifndef OFFPOLICYREINFORCE_H_
#define OFFPOLICYREINFORCE_H_

#include "OffPolicyGradientAlgorithm.h"

namespace ReLe
{

//Templates needed to handle different action types
template<class StateC, class PolicyC, class PolicyC2>
double OffPolicyReinforceIWWorker(const StateC& state, const FiniteAction& action, PolicyC& policy, PolicyC2& behav,
        double& iwb, double& iwt)
{
    typename action_type<FiniteAction>::type_ref u = action.getActionN();

    double valb = behav(state,u);
    double valt = policy(state,u);

    iwt *= valt;
    iwb *= valb;

    return valt/valb;
}

template<class StateC, class ActionC, class PolicyC, class PolicyC2>
double OffPolicyReinforceIWWorker(const StateC& state, const ActionC& action, PolicyC& policy, PolicyC2& behav,
        double& iwb, double& iwt)
{
    double valb = behav(state,action);
    double valt = policy(state,action);

    iwt *= valt;
    iwb *= valb;

    return valt/valb;
}


template<class ActionC, class StateC>
class MBPGA: public AbstractOffPolicyGradientAlgorithm<ActionC, StateC>
{

    USE_PUREOFFPGA_MEMBERS

public:
    MBPGA(DifferentiablePolicy<ActionC, StateC>& target_pol,
          Policy<ActionC, StateC>& behave_pol,
          unsigned int nbEpisodes, unsigned int nbSamplesForJ,
          StepRule& stepL, double penalization = 0.0,
          bool baseline = true, int reward_obj = 0) :
        AbstractOffPolicyGradientAlgorithm<ActionC, StateC>(target_pol, behave_pol, nbEpisodes, nbSamplesForJ, stepL, baseline, reward_obj),
        penal_factor(penalization)
    {
    }

    virtual ~MBPGA()
    {
    }

    inline void setPenalization(double penal)
    {
        penal_factor = penal;
    }

    inline double getPenalization()
    {
        return penal_factor;
    }

protected:
    virtual void init()
    {
        unsigned int dp = target.getParametersSize();
        history_J.assign(nbEpisodesperUpdate,0.0); // policy performance per episode
        history_J_off.assign(nbEpisodesperUpdate,0.0); // policy performance per episode
        history_impWeights.assign(nbEpisodesperUpdate,0.0); // importance weight per episode (w_0 * w_1 * w_2 * ...)
        history_sumdlogpi.assign(nbEpisodesperUpdate,arma::vec(target.getParametersSize())); //gradient log policy per episode (sum)

        bJ_num.zeros(dp); // baseline J
        bM_num.zeros(dp); // baseline M
        b_den.zeros(dp);  // baseline denom (common to J and M)
    }

    virtual void initializeVariables()
    {
        sumdlogpi.zeros();
        prodImpWeightB = 1.0;
        prodImpWeightT = 1.0;
        if (epCounter == 0)
        {
            sumIWOverRun = 0.0;
        }
    }

    virtual double updateStep(const Reward& reward)
    {
        double currIW = OffPolicyReinforceIWWorker(
            currentState, currentAction, target, behavioral, prodImpWeightB, prodImpWeightT);

        arma::vec grad = diffLogWorker(currentState, currentAction, target);
        sumdlogpi += grad;

        return currIW;
    }

    virtual void updateAtEpisodeEnd()
    {
        history_sumdlogpi[epCounter] = sumdlogpi;
        history_impWeights[epCounter] = prodImpWeightT / prodImpWeightB;
        sumIWOverRun += history_impWeights[epCounter];
    }

    virtual void updatePolicy()
    {

        unsigned int dp = target.getParametersSize();
        // expected J value (computed indipendently)
        double Jmean = 0.0;

        // two different set of samples are used to estimate gradient and expected J
        //in_idxs are used to estimate the gradient
        std::vector<unsigned int> in_idxs;
        in_idxs.reserve(nbEpisodesperUpdate-nbIndipendentSamples);
        //out_idxs are used for the expected J
        arma::ivec out_idxs = arma::randi(nbIndipendentSamples, arma::distr_param(0,nbEpisodesperUpdate-1));
        for (int i = 0; i < nbEpisodesperUpdate; ++i)
        {
            arma::uvec q1 = find(out_idxs == i);
            if (q1.is_empty())
            {
                // this data are used for the gradient estimate
                in_idxs.push_back(i);

                //--- update baseline
                arma::vec d2 = history_sumdlogpi[i] % history_sumdlogpi[i];
                bJ_num += history_J[i] * history_impWeights[i] * history_impWeights[i] * d2;
                bM_num += history_J[i] * history_J[i] * history_impWeights[i] * history_impWeights[i] * history_impWeights[i] * d2;
                b_den  += history_impWeights[i] * history_impWeights[i] * d2;
                //---
            }
            else
            {
                // this data are used for the J
                Jmean += history_J_off[i];
            }
        }
        Jmean /= nbIndipendentSamples;

        arma::vec gradientJ(dp, arma::fill::zeros);
        arma::vec gradientM(dp, arma::fill::zeros);
        for (auto i : in_idxs)
        {

            for (int p = 0; p < dp; ++p)
            {

                double baselineJ = 0, baselineM = 0;
                if (useBaseline && b_den[p] != 0)
                {
                    baselineJ = bJ_num[p]/b_den[p];
                    baselineM = bM_num[p]/b_den[p];
                }

                gradientJ[p] += (history_J[i] - baselineJ) * history_impWeights[i] * history_sumdlogpi[i][p];


                gradientM[p] += (history_J[i] * history_J[i] - baselineM / history_impWeights[i]) *
                                history_impWeights[i] * history_impWeights[i] *
                                history_sumdlogpi[i][p];
            }
        }
        //gradientJ /= in_idxs.size();
        //gradientM *= 2.0/in_idxs.size();
        gradientJ /= sumIWOverRun;
        gradientM *= 2.0/sumIWOverRun;


        std::cerr << "gradJ: " << gradientJ.t();
        std::cerr << "gradM: " << gradientM.t();

        arma::vec gradient  = gradientJ
                              - penal_factor * (
                                  gradientM - 2 * Jmean * gradientJ
                              ) / (nbEpisodesperUpdate-nbIndipendentSamples);


        //--- Compute learning step
        arma::mat eMetric = arma::eye(dp,dp);
        arma::vec step_size = stepRule.stepLength(gradient, eMetric);
        //---

        //--- save actual policy performance
        currentItStats->history_J = history_J;
        currentItStats->history_gradients = history_sumdlogpi;
        currentItStats->history_impWeights = history_impWeights;
        currentItStats->estimated_gradient = gradient;
        currentItStats->stepLength = step_size;
        //---


        arma::vec newvalues = target.getParameters() + gradient * step_size;
        target.setParameters(newvalues);
        std::cout << "new_params: "  << newvalues.t();

        for (int i = 0, ie = target.getParametersSize(); i < ie; ++i)
        {
            bJ_num[i] = 0;
            bM_num[i] = 0;
            b_den[i]  = 0;
        }
        sumIWOverRun = 0.0;
    }


protected:
    double penal_factor;
    double prodImpWeightB, prodImpWeightT, sumIWOverRun;
    arma::vec bJ_num, b_den, bM_num;
    arma::vec sumdlogpi;
    std::vector<arma::vec> history_sumdlogpi;
    std::vector<double> history_impWeights;
};


template<class ActionC, class StateC>
class OffpolicyREINFORCE : public MBPGA<ActionC, StateC>
{
public:
    OffpolicyREINFORCE(DifferentiablePolicy<ActionC, StateC>& target,
                       Policy<ActionC, StateC>& behave_pol,
                       unsigned int nbEpisodes, StepRule& stepL,
                       bool baseline = true, int reward_obj = 0)
        : MBPGA<ActionC, StateC>(
            target, behave_pol, nbEpisodes, 0, stepL, 0.0,
            baseline, reward_obj)
    {
    }

    virtual ~OffpolicyREINFORCE()
    {
    }

};

} //end namespace

#endif //OFFPOLICYREINFORCE_H_
