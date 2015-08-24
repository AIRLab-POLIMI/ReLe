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

#include "Agent.h"
#include "Policy.h"
#include "Distribution.h"
#include "BasicFunctions.h"

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
    virtual void initTestEpisode()
    {
    }

    virtual void initEpisode(const StateC &state, ActionC &action)
    {
        sampleAction(state, action);
    }

    void sampleAction(const StateC &state, ActionC &action)
    {
        typename action_type<ActionC>::type_ref u = action;
        u = policy(state);
    }

    void step(const Reward &reward, const StateC &nextState, ActionC &action)
    {
    }

    void endEpisode(const Reward& reward)
    {
    }

    void endEpisode()
    {
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

    virtual void initEpisode(const StateC &state, ActionC &action)
    {
        initTestEpisode();
        this->sampleAction(state, action);
    }

    virtual void initTestEpisode()
    {
        //obtain new parameters
        arma::vec new_params = dist();
        //set to policy
        policy.setParameters(new_params);
    }

    virtual ~PolicyEvalDistribution()
    {

    }

private:
    ParametricPolicy<ActionC, StateC>& policy;
    Distribution& dist;
};

} //end namespace

#endif //POLICYEVALAGENT_H_
