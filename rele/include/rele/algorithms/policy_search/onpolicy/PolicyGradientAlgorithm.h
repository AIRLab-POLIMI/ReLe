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

#ifndef POLICYGRADIENTALGORITHM_H_
#define POLICYGRADIENTALGORITHM_H_

#include "Agent.h"
#include "Policy.h"
#include "Basics.h"
#include "BasicFunctions.h"
#include "policy_search/onpolicy/GradientOutputData.h"
#include <cassert>
#include <iomanip>

namespace ReLe
{

////Templates needed to sample different action types
template<class StateC, class PolicyC>
arma::vec diffLogWorker(const StateC& state, FiniteAction& action, PolicyC& policy)
{
    unsigned int u = action.getActionN();
    return policy.difflog(state, u);
}

template<class StateC, class ActionC, class PolicyC>
arma::vec diffLogWorker(const StateC& state, ActionC& action, PolicyC& policy)
{
    return policy.difflog(state, action);
}

template<class ActionC, class StateC>
class PolicyGradientAlgorithm: public Agent<ActionC, StateC>
{

public:
    PolicyGradientAlgorithm(DifferentiablePolicy<ActionC, StateC>& policy,
                            unsigned int nbEpisodes, double stepL,
                            bool baseline = true, int reward_obj = 0) :
        policy(policy), nbEpisodesToEvalPolicy(nbEpisodes),
        runCount(0), epiCount(0), df(1.0), Jep(0.0), rewardId(reward_obj),
        useBaseline(baseline), output2LogReady(false), stepLength(stepL),
        currentItStats(nullptr)
    {
    }

    virtual ~PolicyGradientAlgorithm()
    {
    }

    // Agent interface
public:
    virtual void initEpisode(const StateC& state, ActionC& action)
    {
        df = 1.0;    //reset discount factor
        Jep = 0.0;    //reset J of current episode

        // Initialize variables
        sumdlogpi.zeros();

        //--- set up agent output
        if (epiCount == 0)
        {
            currentItStats = new GradientIndividual();
            currentItStats->policy_parameters = policy.getParametersSize();
        }
        //---

        sampleAction(state, action);

        // save state and action for late use
        currentState  = state;
        currentAction = action;
    }

    virtual void initTestEpisode()
    {
    }

    virtual void sampleAction(const StateC& state, ActionC& action)
    {
        sampleActionWorker(state, action, policy);
    }

    virtual void step(const Reward& reward, const StateC& nextState,
                      ActionC& action)
    {
        arma::vec grad = diffLogWorker(currentState, currentAction, policy);
        sumdlogpi += grad;

        //calculate current J value
        Jep += df * reward[rewardId];
        //update discount factor
        df *= this->task.gamma;

        sampleAction(nextState, action);

        // save state and action for late use
        currentState  = nextState;
        currentAction = action;
    }

    virtual void endEpisode(const Reward& reward)
    {
        arma::vec grad = diffLogWorker(currentState, currentAction, policy);
        sumdlogpi += grad;

        //add last contribute
        Jep += df * reward[rewardId];

        //perform remaining operation
        this->endEpisode();

    }

    virtual void endEpisode()
    {

        //save policy value
        history_J[epiCount] = Jep;

        for (int p = 0; p < baseline_num.n_elem; ++p)
        {
            baseline_num[p] += Jep * sumdlogpi[p] * sumdlogpi[p];
            baseline_den[p] += sumdlogpi[p] * sumdlogpi[p];
            history_sumdlogpi[epiCount][p] = sumdlogpi[p];
        }

        //last episode is the number epiCount+1
        epiCount++;
        //check evaluation of actual policy
        if (epiCount == nbEpisodesToEvalPolicy)
        {
            updatePolicy();

            //reset counters and gradient
            epiCount = 0; //reset episode counter
            runCount++; //update run counter
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
        history_J.assign(nbEpisodesToEvalPolicy, 0.0);
        history_sumdlogpi.assign(nbEpisodesToEvalPolicy,arma::vec(policy.getParametersSize()));

        baseline_den.zeros(policy.getParametersSize());
        baseline_num.zeros(policy.getParametersSize());

        sumdlogpi.set_size(policy.getParametersSize());
    }

    virtual void updatePolicy()
    {
        int nbParams = baseline_num.n_elem;
        arma::vec gradient(nbParams, arma::fill::zeros);
        // In the previous loop I have computed the baseline
        for (int ep = 0; ep < nbEpisodesToEvalPolicy; ++ep)
        {
            for (int p = 0; p < nbParams; ++p)
            {
                double base_el = (baseline_den[p] == 0.0 || useBaseline == false) ? 0 : baseline_num[p]/baseline_den[p];
                gradient[p] += history_sumdlogpi[ep][p] * (history_J[ep] - base_el);
            }
        }

        //--- Compute learning step
        //http://www.ias.informatik.tu-darmstadt.de/uploads/Geri/lecture-notes-constraint.pdf
        double lambda = arma::dot(gradient,gradient) / (4*stepLength);
        lambda = sqrt(lambda);
        lambda = std::max(lambda, 1e-8); // to avoid numerical problems
        double step_size = 1.0 / (2 * lambda);
        //        std::cout << "step_size: " << step_size << std::endl;
        //---

        //--- save actual policy performance
        currentItStats->history_J = history_J;
        currentItStats->history_gradients = history_sumdlogpi;
        currentItStats->estimated_gradient = gradient;
        currentItStats->stepLength = step_size;
        //---


        arma::vec newvalues = policy.getParameters() + gradient * step_size;
        policy.setParameters(newvalues);
        //        std::cout << "new_params: "  << newvalues.t();

        for (int i = 0; i < nbParams; ++i)
        {
            baseline_den[i] = 0;
            baseline_num[i] = 0;
        }
    }

protected:
    DifferentiablePolicy<ActionC, StateC>& policy;
    unsigned int nbEpisodesToEvalPolicy;
    unsigned int runCount, epiCount;
    double df, Jep, stepLength;
    int rewardId;

    arma::vec sumdlogpi, baseline_den, baseline_num;
    std::vector<double> history_J;
    std::vector<arma::vec> history_sumdlogpi;

    bool useBaseline, output2LogReady;


    GradientIndividual* currentItStats;

    ActionC currentAction;
    StateC currentState;
};

}// end namespace ReLe

#endif //POLICYGRADIENTALGORITHM_H_
