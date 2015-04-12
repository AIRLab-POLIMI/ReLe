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

#ifndef OFFALGORITHM_H_
#define OFFALGORITHM_H_

#include "Agent.h"
#include "Distribution.h"
#include "Policy.h"
#include "Basics.h"
#include "BasicFunctions.h"
#include <cassert>
#include <iomanip>
#include "policy_search/offpolicy/OffGradientOutputData.h"

namespace ReLe
{

//Templates needed to handle different action types
template<class StateC, class PolicyC, class PolicyC2>
double PureOffAlgorithmStepWorker(const StateC& state, const FiniteAction& action, PolicyC& policy, PolicyC2& behav,
                                  double& iwb, double& iwt, arma::vec& grad)
{
    typename action_type<FiniteAction>::type_ref u = action.getActionN();

    double valb = behav(state,u);
    double valt = policy(state,u);

    iwt *= valt;
    iwb *= valb;

    //init the sum of the gradient of the policy logarithm
    arma::vec logGradient = policy.difflog(state, u);
    grad += logGradient;

    return valt/valb;
}

template<class StateC, class ActionC, class PolicyC, class PolicyC2>
double PureOffAlgorithmStepWorker(const StateC& state, const ActionC& action, PolicyC& policy, PolicyC2& behav,
                                  double& iwb, double& iwt, arma::vec& grad)
{
    double valb = behav(state,action);
    double valt = policy(state,action);

    iwt *= valt;
    iwb *= valb;

    //init the sum of the gradient of the policy logarithm
    arma::vec logGradient = policy.difflog(state, action);
    grad += logGradient;

    return valt/valb;
}


template<class ActionC, class StateC>
class PureOffAlgorithm: public BatchAgent<ActionC, StateC>
{

public:
    PureOffAlgorithm(DifferentiablePolicy<ActionC, StateC>& target_pol,
                     Policy<ActionC, StateC>& behave_pol,
                     unsigned int nbPolicies, unsigned int nbSamplesForJandVar,
                     double penalization = 0.0, double stepL = 0.5,
                     bool baseline = true, int reward_obj = 0) :
        target(target_pol), behavioral(behave_pol), nbEpisodesperUpdate(nbPolicies), runCounter(0),
        epCounter(0), df(1.0), Jep(0.0), rewardId(reward_obj),
        useBaseline(baseline), output2LogReady(false),
        currentItStats(nullptr), stepLength(stepL), penal_factor(penalization),
        nbIndipendentSamples(std::min(std::max(1,static_cast<int>(nbSamplesForJandVar)), static_cast<int>(nbPolicies*0.5)))
    {
        prodImpWeightT = 1.0;
        prodImpWeightB = 1.0;
    }

    virtual ~PureOffAlgorithm()
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

    inline void setStepLength(double penal)
    {
        stepLength = penal;
    }

    inline double getStepLength()
    {
        return stepLength;
    }

    // Agent interface
public:
    virtual void initEpisode(const StateC& state, const ActionC& action)
    {
        df  = 1.0;    //reset discount factor
        Jep = 0.0;    //reset J of current episode
        Jepoff = 0.0;

        //--- set up agent output
        if (epCounter == 0)
        {
            currentItStats = new OffGradientIndividual();
            currentItStats->policy_parameters = target.getParameters();
        }
        //---

        prodImpWeightB = 1.0;
        prodImpWeightT = 1.0;
        sumdlogpi.zeros(target.getParametersSize());
        currentState  = state;
        currentAction = action;

        //        std::cout << std::endl;
    }

    virtual void initTestEpisode()
    {
    }

    virtual void sampleAction(const StateC& state, ActionC& action)
    {
        sampleActionWorker(state, action, target);
    }

    virtual void step(const Reward& reward, const StateC& nextState,
                      const ActionC& nextAction)
    {


        //        std::cout << currentState << " " << currentAction << " " << reward[0] << std::endl;

        double currentIW = PureOffAlgorithmStepWorker(currentState, currentAction, target, behavioral, prodImpWeightB, prodImpWeightT, sumdlogpi);

        //calculate current J value
        Jep += df * reward[rewardId];
        Jepoff += df * currentIW * reward[rewardId];
        //update discount factor
        df *= this->task.gamma;

        currentState  = nextState;
        currentAction = nextAction;
    }

    virtual void endEpisode(const Reward& reward)
    {


        //        std::cout << currentState << " " << currentAction << " " << reward[0] << std::endl;

        double currentIW = PureOffAlgorithmStepWorker(currentState, currentAction, target, behavioral, prodImpWeightB, prodImpWeightT, sumdlogpi);

        //add last contribute
        Jep += df * reward[rewardId];
        Jepoff += df * currentIW * reward[rewardId];
        //perform remaining operation
        this->endEpisode();

    }

    virtual void endEpisode()
    {

        history_J[epCounter] = Jep;
        history_J_off[epCounter] = Jepoff;
        history_impWeights[epCounter] = prodImpWeightT / prodImpWeightB;
        history_sumdlogpi[epCounter] = sumdlogpi;
        sumIWOverRun += history_impWeights[epCounter];

        ++epCounter;

        if (epCounter == nbEpisodesperUpdate)
        {
            //all policies have been evaluated
            //conclude gradient estimate and update the distribution
            updatePolicy();

            //reset counters and gradient
            epCounter = 0; //reset policy counter
            runCounter++; //update run counter
            output2LogReady = true; //output must be ready for log
        }
    }

    virtual AgentOutputData* getAgentOutputDataEnd()
    {
        if (output2LogReady)
        {
            //output is ready, activate flag
            output2LogReady = false;
            return currentItStats;
        }
        return nullptr;
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
        //http://www.ias.informatik.tu-darmstadt.de/uploads/Geri/lecture-notes-constraint.pdf
        double lambda = arma::dot(gradient,gradient) / (4*stepLength);
        lambda = sqrt(lambda);
        lambda = std::max(lambda, 1e-8); // to avoid numerical problems
        double step_size = 1.0 / (2 * lambda);
        std::cout << "step_size: " << step_size << std::endl;
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
    DifferentiablePolicy<ActionC, StateC>& target;
    Policy<ActionC, StateC>& behavioral;
    unsigned int nbEpisodesperUpdate;
    unsigned int runCounter, epCounter;
    double df, Jep, Jepoff, stepLength, penal_factor, sumIWOverRun;
    int rewardId;

    double prodImpWeightB, prodImpWeightT;
    arma::vec sumdlogpi, bJ_num, b_den, bM_num;

    std::vector<double> history_J;
    std::vector<double> history_J_off;
    std::vector<double> history_impWeights;
    std::vector<arma::vec> history_sumdlogpi;

    bool useBaseline, output2LogReady;
    OffGradientIndividual* currentItStats;

    unsigned int nbIndipendentSamples;
    ActionC currentAction;
    StateC currentState;
};


template<class ActionC, class StateC>
class OffpolicyREINFORCE : public PureOffAlgorithm<ActionC, StateC>
{
public:
    OffpolicyREINFORCE(DifferentiablePolicy<ActionC, StateC>& target,
                       Policy<ActionC, StateC>& behave_pol,
                       unsigned int nbPolicies, double stepL = 0.5,
                       bool baseline = true, int reward_obj = 0)
        : PureOffAlgorithm<ActionC, StateC>(
            target, behave_pol, nbPolicies, 0, 0.0, stepL,
            baseline, reward_obj)
    {
    }

    virtual ~OffpolicyREINFORCE()
    {
    }

};

} //end namespace

#endif //OFFALGORITHM_H_
