/*
 * rele,
 *
 *
 * Copyright (C) 2015 Davide Tateo & Matteo Pirotta & Marcello Restelli
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

#ifndef POLICYEVALAGENT_H_
#define POLICYEVALAGENT_H_

#include "rele/core/Agent.h"
#include "rele/policy/Policy.h"
#include "rele/statistics/Distribution.h"
#include "rele/core/BasicFunctions.h"

namespace ReLe
{

template<class ActionC, class StateC>
class PolicyEvalAgent: public Agent<ActionC, StateC>
{
public:
    PolicyEvalAgent(Policy<ActionC, StateC>& policy): policy(policy)
    {

    }

    virtual ~PolicyEvalAgent()
    {
    }

    // Agent interface
public:
    virtual void initTestEpisode() override
    {
    }

    virtual void initEpisode(const StateC &state, ActionC &action) override
    {
        sampleAction(state, action);
    }

    void sampleAction(const StateC &state, ActionC &action) override
    {
        typename action_type<ActionC>::type_ref u = action;
        u = policy(state);
    }

    void step(const Reward &reward, const StateC &nextState, ActionC &action) override
    {
        throw std::runtime_error("PolicyEvalAgent cannot be used for learning!");
    }

    void endEpisode(const Reward& reward) override
    {
        throw std::runtime_error("PolicyEvalAgent cannot be used for learning!");
    }

    void endEpisode() override
    {
        throw std::runtime_error("PolicyEvalAgent cannot be used for learning!");
    }

private:
    Policy<ActionC, StateC>& policy;

};

template<class ActionC, class StateC>
class PolicyEvalDistribution : public PolicyEvalAgent<ActionC, StateC>
{
public:

    PolicyEvalDistribution(Distribution& dist, ParametricPolicy<ActionC, StateC>& policy)
        : PolicyEvalAgent<ActionC, StateC>(policy), policy(policy), dist(dist)
    {

    }

    virtual void initEpisode(const StateC &state, ActionC &action) override
    {
        initTestEpisode();
        this->sampleAction(state, action);
    }

    virtual void initTestEpisode() override
    {
        //obtain new parameters
        arma::vec new_params = dist();

        //Save them in the history
        params_history.push_back(new_params);

        //set to policy
        policy.setParameters(new_params);
    }

    arma::mat getParams()
    {
        arma::mat params(policy.getParametersSize(), params_history.size());

        for(int i = 0; i < params.n_cols; i++)
        {
            params.col(i) = params_history[i];
        }

        return params;
    }


    virtual ~PolicyEvalDistribution()
    {

    }

private:
    ParametricPolicy<ActionC, StateC>& policy;
    Distribution& dist;
    std::vector<arma::vec> params_history;
};

} //end namespace

#endif //POLICYEVALAGENT_H_
