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

#include "Approximators.h"

#include <armadillo>
#include "../core/Environment.h"

namespace ReLe
{

template<class StateC>
arma::vec computeRewardWorker(const StateC& state, ParametricRegressor& regressor)
{
    //TODO check this. probably is not correct!
    return regressor(state);
}

template<>
arma::vec computeRewardWorker<FiniteState>(const FiniteState& state, ParametricRegressor& regressor)
{
    arma::vec input(1);
    input[0] = state.getStateN();
    return regressor(input);
}

template<class ActionC, class StateC>
class ParametricRewardMDP : public Environment<ActionC, StateC>
{

public:
    ParametricRewardMDP(Environment<ActionC, StateC>& envirorment,
                        ParametricRegressor& regressor) :
        envirorment(envirorment), regressor(regressor)
    {
        this->getWritableSettings() = envirorment.getSettings();
        this->getWritableSettings().rewardDim = regressor.getOutputSize();
    }

    virtual void step(const ActionC& action, StateC& nextState,
                      Reward& reward)
    {
        Reward trashReward(envirorment.getSettings().rewardDim);
        envirorment.step(action, nextState, trashReward);

        arma::vec&& r = computeReward(nextState);
        reward = arma::conv_to<Reward>::from(r);
    }

    virtual void getInitialState(StateC& state)
    {
        envirorment.getInitialState(state);
    }

    virtual ~ParametricRewardMDP()
    {

    }


private:
    arma::vec computeReward(StateC& nextState)
    {
        return computeRewardWorker(nextState, regressor);
    }

private:
    Environment<ActionC, StateC>& envirorment;
    ParametricRegressor& regressor;
};


}



#endif /* INCLUDE_RELE_IRL_PARAMETRICREWARDMDP_H_ */
