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

#ifndef INCLUDE_RELE_ENVIRONMENTS_MAB_MAB_H_
#define INCLUDE_RELE_ENVIRONMENTS_MAB_MAB_H_

#include "Environment.h"

namespace ReLe
{

template<class ActionC>
class MAB: public Environment<ActionC, FiniteState>
{
public:
    MAB(double gamma, unsigned int horizon = 1) : Environment<ActionC, FiniteState>()
    {
        EnvironmentSettings& task = this->getWritableSettings();
        task.isFiniteHorizon = true;
        task.horizon = horizon;
        task.gamma = gamma;
        task.isAverageReward = false;
        task.isEpisodic = true;
        task.finiteStateDim = 1;
        task.continuosStateDim = 0;
        task.rewardDim = 1;
    }

    void getInitialState(FiniteState& state) override
    {
        state.setStateN(0);
        state.setAbsorbing(false);
    }
};

}

#endif /* INCLUDE_RELE_ENVIRONMENTS_MAB_MAB_H_ */
