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

namespace ReLe
{

template<class ActionC, class StateC, class PolicyC>
class PolicyEvalAgent: public Agent<ActionC, StateC>
{
public:
    PolicyEvalAgent(PolicyC& policy): policy(policy)
    {
    }
    virtual ~PolicyEvalAgent()
    {
    }

    // Agent interface
public:
    void initTestEpisode()
    {
    }

    void initEpisode(const StateC &state, ActionC &action)
    {
    }

    void sampleAction(const StateC &state, ActionC &action)
    {
        typename action_type<ActionC>::type_ref u = action;
        u = policy(state);
    }

    template<class FiniteAction>
    void sampleAction(const StateC& state, FiniteAction& action)
    {
        unsigned int u = policy(state);
        action.setActionN(u);
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
    PolicyC policy;
};

} //end namespace

#endif //POLICYEVALAGENT_H_
