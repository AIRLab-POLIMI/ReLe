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

#ifndef INCLUDE_RELE_IRL_PARAMETRICREWARDMDP_H_
#define INCLUDE_RELE_IRL_PARAMETRICREWARDMDP_H_

#include <armadillo>

#include "rele/approximators/Regressors.h"
#include "rele/core/Environment.h"

#include "rele/core/BasicFunctions.h"

namespace ReLe
{

template<class ActionC, class StateC>
class ParametricRewardMDP : public Environment<ActionC, StateC>
{

public:
    ParametricRewardMDP(Environment<ActionC, StateC>& environment,
                        ParametricRegressor& regressor) :
        environment(environment), regressor(regressor)
    {
        this->getWritableSettings() = environment.getSettings();
        this->getWritableSettings().rewardDimensionality = regressor.getOutputSize();
    }

    virtual void step(const ActionC& action, StateC& nextState,
                      Reward& reward) override
    {
        Reward trashReward(environment.getSettings().rewardDimensionality);
        environment.step(action, nextState, trashReward);

        arma::vec&& r = computeReward(state, action, nextState);
        reward = arma::conv_to<Reward>::from(r);

        state = nextState;
    }

    virtual void getInitialState(StateC& state) override
    {
        environment.getInitialState(state);
        this->state = state;
    }

    virtual ~ParametricRewardMDP()
    {

    }


private:
    arma::vec computeReward(const StateC& state, const ActionC& action, const StateC& nextState)
    {
        return regressor(state, action, nextState);
    }

private:
    Environment<ActionC, StateC>& environment;
    ParametricRegressor& regressor;
    StateC state;
};


}



#endif /* INCLUDE_RELE_IRL_PARAMETRICREWARDMDP_H_ */
