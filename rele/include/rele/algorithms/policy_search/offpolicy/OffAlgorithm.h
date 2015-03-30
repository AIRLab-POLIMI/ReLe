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
double PureOffAlgorithmComputeIWWorker(const StateC& state, const FiniteAction& action, PolicyC& policy, PolicyC2& behav)
{
    typename action_type<FiniteAction>::type_ref u = action.getActionN();
    return policy(state,u) / behav(state,u);
}

template<class StateC, class ActionC, class PolicyC, class PolicyC2>
double PureOffAlgorithmComputeIWWorker(const StateC& state, const ActionC& action, PolicyC& policy, PolicyC2& behav)
{
    return policy(state,action) / behav(state,action);
}

template<class StateC, class PolicyC, class PolicyC2>
double PureOffAlgorithmStepWorker(const StateC& state, const FiniteAction& action, PolicyC& policy, PolicyC2& behav,
                                  double& iw, arma::vec& grad)
{
    typename action_type<FiniteAction>::type_ref u = action.getActionN();

    double val = policy(state,u) / behav(state,u);
    iw *= val;

    //init the sum of the gradient of the policy logarithm
    arma::vec logGradient = policy.difflog(state, u);
    grad += logGradient;

}

template<class StateC, class ActionC, class PolicyC, class PolicyC2>
double PureOffAlgorithmStepWorker(const StateC& state, const ActionC& action, PolicyC& policy, PolicyC2& behav,
                                  double& iw, arma::vec& grad)
{
    double val = policy(state,action) / behav(state,action);
    iw *= val;

    //init the sum of the gradient of the policy logarithm
    arma::vec logGradient = policy.difflog(state, action);
    grad += logGradient;

    return val;
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
        prodImpWeight = 1.0;
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

        //        prodImpWeight = policy(state,action) / behavioral(state,action);

        //        //init the sum of the gradient of the policy logarithm
        //        arma::vec logGradient = policy.difflog(state, action);
        //        sumdlogpi = logGradient;

        //--- set up agent output
        currentItStats = new OffGradientIndividual();
        //---

        prodImpWeight = 1.0;
        sumdlogpi.zeros(target.getParametersSize());
        currentIW = PureOffAlgorithmStepWorker(state,action, target, behavioral, prodImpWeight, sumdlogpi);
    }

    virtual void initTestEpisode()
    {
    }

    virtual void sampleAction(const StateC& state, ActionC& action)
    {
        sampleActionWorker(state, action, target);
    }

    virtual void step(const Reward& reward, const StateC& nextState,
                      const ActionC& action)
    {
        //calculate current J value
        Jep += df * currentIW * reward[rewardId];
        //update discount factor
        df *= this->task.gamma;

        //        //update importance sampling
        //        double ival = policy(nextState,action) / behavioral(nextState,action);
        //        prodImpWeight *= ival;

        //        //update sum of the gradient of the policy logarithm
        //        arma::vec logGradient = policy.difflog(nextState, action);
        //        sumdlogpi += logGradient;

        currentIW = PureOffAlgorithmStepWorker(nextState, action, target, behavioral, prodImpWeight, sumdlogpi);
    }

    virtual void endEpisode(const Reward& reward)
    {
        //add last contribute
        Jep += df * currentIW * reward[rewardId];
        //perform remaining operation
        this->endEpisode();

    }

    virtual void endEpisode()
    {

        history_J[epCounter] = Jep;
        history_impWeights[epCounter] = prodImpWeight;
        history_sumdlogpi[epCounter] = sumdlogpi;

        //        //--- update baseline (moved down in the update)
        //        arma::vec d2 = sumdlogpi % sumdlogpi;
        //        bJ_num += Jep * prodImpWeight * prodImpWeight * d2;
        //        bJ_den += prodImpWeight * prodImpWeight * d2;


        //        bM_num += Jep * Jep * prodImpWeight * prodImpWeight * prodImpWeight * d2;
        //        bM_den += prodImpWeight * prodImpWeight * d2;
        //        //---

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
        history_J.assign(nbEpisodesperUpdate,0.0);
        history_impWeights.assign(nbEpisodesperUpdate,0.0);
        history_sumdlogpi.assign(nbEpisodesperUpdate,arma::vec(target.getParametersSize()));

        bJ_num.zeros(target.getParametersSize());
        bJ_den.zeros(target.getParametersSize());
        bM_num.zeros(target.getParametersSize());
        bM_den.zeros(target.getParametersSize());
    }

    virtual void updatePolicy()
    {

        double Jmean = 0.0, Jvar = 0.0;

        std::vector<unsigned int> in_idxs;
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
                bJ_den += history_impWeights[i] * history_impWeights[i] * d2;


                bM_num += history_J[i] * history_J[i] * history_impWeights[i] * history_impWeights[i] * history_impWeights[i] * d2;
                bM_den += history_impWeights[i] * history_impWeights[i] * d2;
                //---
            }
            else
            {
                // this data are used for the J and Var(J) estimate
                Jmean += history_J[i];
            }
        }
        Jmean /= nbIndipendentSamples;

        arma::vec baselineJ(target.getParametersSize(), arma::fill::zeros);
        arma::vec baselineM(target.getParametersSize(), arma::fill::zeros);
        if (useBaseline)
        {
            for (int i = 0; i < bJ_den.n_elem; ++i)
            {
                baselineJ[i] = bJ_den[0] != 0 ? bJ_num[i]/bJ_den[i] : 0.0;
                baselineM[i] = bM_den[0] != 0 ? bM_num[i]/bM_den[i] : 0.0;
            }
        }

        arma::vec gradientJ(target.getParametersSize(), arma::fill::zeros);
        arma::vec gradientM(target.getParametersSize(), arma::fill::zeros);
        for (auto i : in_idxs)
        {

            //            std::cout << history_impWeights[i] << " " << history_J[i] << std::endl;
            //            std::cout << history_sumdlogpi[i].t();
            gradientJ += history_impWeights[i] * history_J[i] * (history_sumdlogpi[i] - baselineJ);
            //            if (isnan(gradientJ[0]))
            //            {
            //                std::cout << "error" << std::endl;
            //            }
            //            std::cout << gradientJ.t();


            gradientM += history_impWeights[i] * history_impWeights[i] *
                         history_J[i] * history_J[i] *
                         (history_sumdlogpi[i] - baselineM);
        }
        gradientJ /= in_idxs.size();
        gradientM *= 2.0/in_idxs.size();


        for (auto i : out_idxs)
        {
            Jvar += (history_J[i] - Jmean) * (history_J[i] - Jmean);
        }
        Jvar /= nbIndipendentSamples;
        double stddevJ = std::max(1e-8,sqrt(Jvar)); // to avoid numerical problems

        std::cerr << gradientJ.t();

        arma::vec gradient  = gradientJ
                              + penal_factor * (
                                  gradientM - 2 * Jmean * gradientJ
                              ) / (2 * (nbEpisodesperUpdate-nbIndipendentSamples) * stddevJ);


        //--- Compute learning step
        //http://www.ias.informatik.tu-darmstadt.de/uploads/Geri/lecture-notes-constraint.pdf
        double lambda = arma::dot(gradient,gradient) / (4*stepLength);
        lambda = sqrt(lambda);
        lambda = std::max(lambda, 1e-8); // to avoid numerical problems
        double step_size = 1.0 / (2 * lambda);
        std::cout << step_size << std::endl;
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


        //TODO ciclo unico
        for (int i = 0, ie = target.getParametersSize(); i < ie; ++i)
        {
            bJ_num[i] = 0;
            bJ_den[i] = 0;
            bM_num[i] = 0;
            bM_den[i] = 0;
        }
    }


protected:
    DifferentiablePolicy<ActionC, StateC>& target;
    Policy<ActionC, StateC>& behavioral;
    unsigned int nbEpisodesperUpdate;
    unsigned int runCounter, epCounter;
    double df, Jep, stepLength, penal_factor;
    int rewardId;

    double prodImpWeight, currentIW;
    arma::vec sumdlogpi, bJ_num, bJ_den, bM_num, bM_den;

    std::vector<double> history_J;
    std::vector<double> history_impWeights;
    std::vector<arma::vec> history_sumdlogpi;

    bool useBaseline, output2LogReady;
    OffGradientIndividual* currentItStats;

    unsigned int nbIndipendentSamples;
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
