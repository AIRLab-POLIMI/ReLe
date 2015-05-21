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

#include "MultiHeat.h"
#include "RandomGenerator.h"

using namespace std;

namespace ReLe
{

MultiHeatSettings::MultiHeatSettings()
{
    MultiHeatSettings::defaultSettings(*this);
}

void MultiHeatSettings::defaultSettings(MultiHeatSettings& settings)
{
    //Environment Parameters
    settings.gamma = 0.99;
    settings.continuosStateDim = 3;
    settings.continuosActionDim = 1;
    settings.rewardDim = 1;
    settings.finiteStateDim = 0;
    settings.finiteActionDim = 0;
    settings.isFiniteHorizon = false;
    settings.isAverageReward = false;
    settings.isEpisodic = true;
    settings.horizon = 300;

    //MultiHeat Parameters
    settings.Nr;
    settings.Ta;
    settings.dt;
    settings.a;
    settings.s2n;

    settings.A.zeros(settings.Nr,settings.Nr);
    settings.A(0,0) = 4;
    settings.B;
    settings.C;

    settings.TUB;
    settings.TLB;
}

MultiHeatSettings::~MultiHeatSettings()
{

}

void MultiHeatSettings::WriteToStream(std::ostream& out) const
{
    //TODO implement
}

void MultiHeatSettings::ReadFromStream(std::istream& in)
{
    //TODO implement
}

//=================================================================
// Multi Heat MDP
//-----------------------------------------------------------------

MultiHeat::MultiHeat() :
    DenseMDP(new MultiHeatSettings()), cleanConfig(true)
{
    config = static_cast<MultiHeatSettings*>(settings);
    currentState.set_size(config->continuosStateDim);

    //initialize transition matrix
    computeTransitionMatrix();
}

MultiHeat::MultiHeat(MultiHeatSettings& config)
    : DenseMDP(&config), cleanConfig(false), config(&config)
{
    currentState.set_size(this->getSettings().continuosStateDim);

    //initialize transition matrix
    computeTransitionMatrix();
}

void MultiHeat::step(const FiniteAction& action,
                     DenseState& nextState, Reward& reward)
{


    nextState = currentState;
}

void MultiHeat::getInitialState(DenseState& state)
{
    currentState.setAbsorbing(false);
    currentState(0) = 4;
    state = currentState;
}

void MultiHeat::computeTransitionMatrix()
{
    Xi(9) = config->s2n * config->dt;
}

}
