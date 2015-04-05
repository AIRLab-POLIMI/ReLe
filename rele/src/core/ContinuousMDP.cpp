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

#include "ContinuousMDP.h"

using namespace std;

namespace ReLe
{

ContinuousMDP::ContinuousMDP(size_t stateSize, size_t actionSize, size_t rewardSize, bool isFiniteHorizon,
                             bool isEpisodic, double gamma, unsigned int horizon) :
    currentState(stateSize)
{
    setupEnvirorment(stateSize, actionSize, rewardSize, isFiniteHorizon, isEpisodic, horizon,
                     gamma);
}

void ContinuousMDP::setupEnvirorment(size_t stateSize, size_t actionSize, size_t rewardSize,
                                     bool isFiniteHorizon, bool isEpisodic, unsigned int horizon,
                                     double gamma)
{
    EnvironmentSettings& task = getWritableSettings();
    task.isFiniteHorizon = isFiniteHorizon;
    task.horizon = horizon;
    task.gamma = gamma;
    task.isAverageReward = false;
    task.isEpisodic = isEpisodic;
    task.finiteStateDim = 0;
    task.finiteActionDim = 0;
    task.continuosStateDim = stateSize;
    task.continuosActionDim = actionSize;
    task.rewardDim = rewardSize;
}

}
