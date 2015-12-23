/*
 * rele,
 *
 *
 * Copyright (C) 2015 Francesco Trovò and Stefano Paladino
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

#ifndef MABALGORITHM_H_
#define MABALGORITHM_H_

#include "Agent.h"
#include "Basics.h"


namespace ReLe
{

template<class ActionC, class StateC>
class MABAlgorithm: public Agent<ActionC, StateC>
{

public:
    MABAlgorithm(ParametricPolicy<ActionC, StateC>& policy):
        policy(policy)
    {

    }

    virtual ~MABAlgorithm()
    {

    }

public:
    /**
     *
     * It does not make sense to ask to a MAB algorithm for the optimal
     * action, since we are in an online learning paradigm
     *
     */
    virtual void initTestEpisode() override
    {
        arma::vec&& parameters = selectNextArm();
        policy.setParameters(parameters);
    }

    virtual void initEpisode(const StateC& state, ActionC& action) override
    {
        arma::vec&& parameters = selectNextArm();
        policy.setParameters(parameters);
        sampleAction(state,action);
    }

    virtual void sampleAction(const StateC& state, ActionC& action) override
    {
        typename action_type<ActionC>::type_ref u = action;
        u = policy(state);
    }

    virtual void step(const Reward& reward, const StateC& nextState,
                      ActionC& action) override
    {
        updateHistory(reward);
        sampleAction(nextState,action);
    }

    virtual void endEpisode(const Reward& reward) override
    {
        updateHistory(reward);
    }

    virtual void endEpisode() override
    {
    }

protected:
    virtual arma::vec selectNextArm() = 0;
    virtual void updateHistory(const Reward& reward) = 0;

protected:
    ParametricPolicy<ActionC, StateC>& policy;
};

}

#endif /* MABALGORITHM_H_ */
