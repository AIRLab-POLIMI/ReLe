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

#include "rele/environments/MultiHeat.h"
#include "rele/utils/RandomGenerator.h"

using namespace std;
using namespace arma;

namespace ReLe
{

MultiHeatSettings::MultiHeatSettings()
{
    MultiHeatSettings::defaultSettings(*this);
}

void MultiHeatSettings::defaultSettings(MultiHeatSettings& settings)
{
    //MultiHeat Parameters
    settings.Nr = 2;
    settings.Ta = 6;
    settings.dt = 0.1;
    settings.a = 0.8;
    settings.s2n = 1;

    settings.A.zeros(settings.Nr,settings.Nr);
    settings.A.diag(1).fill(0.33*settings.dt);
    settings.A.diag(-1).fill(0.33*settings.dt);
    settings.B.zeros(settings.Nr,1);
    settings.B.fill(0.25*settings.dt);
    settings.C.zeros(settings.Nr,1);
    settings.C.fill(12*settings.dt);

    settings.TUB = 22;
    settings.TLB = 17.5;


    //Environment Parameters
    settings.gamma = 0.95;
    settings.stateDimensionality = settings.Nr+1;
    settings.actionDimensionality = 0;
    settings.rewardDimensionality = 1;
    settings.statesNumber = 0;
    settings.actionsNumber = settings.Nr+1;
    settings.isFiniteHorizon = false;
    settings.isAverageReward = false;
    settings.isEpisodic = false;
    settings.horizon = 300;
}

MultiHeatSettings::~MultiHeatSettings()
{

}

void MultiHeatSettings::WriteToStream(std::ostream& out) const
{
    //TODO [SERIALIZATION] implement
}

void MultiHeatSettings::ReadFromStream(std::istream& in)
{
    //TODO [SERIALIZATION] implement
}

//=================================================================
// Multi Heat MDP
//-----------------------------------------------------------------

MultiHeat::MultiHeat() :
    DenseMDP(new MultiHeatSettings()), cleanConfig(true)
{
    config = static_cast<MultiHeatSettings*>(settings);
    currentState.set_size(config->stateDimensionality);

    //initialize transition matrix
    computeTransitionMatrix();
}

MultiHeat::MultiHeat(MultiHeatSettings& config)
    : DenseMDP(&config), cleanConfig(false), config(&config)
{
    currentState.set_size(this->getSettings().stateDimensionality);

    //initialize transition matrix
    computeTransitionMatrix();
}

void MultiHeat::step(const FiniteAction& action,
                     DenseState& nextState, Reward& reward)
{
    // Mode Transition
    double randExtr = RandomGenerator::sampleUniform(0, 1);
    bool success = (randExtr <= config->a) ? true : false;
    if (action != currentState(mode) && success)
    {
        currentState(mode) = action;
    }

    // Continuous State Transition
    vec noise = sqrt(config->s2n*config->dt)*randn<vec>(config->Nr);
    currentState.rows(1,config->Nr) += Xi*currentState.rows(1,config->Nr)+Gamma + noise;
    if (currentState(mode) > 0)
    {
        currentState(currentState(mode)) += config->C(currentState(mode)-1);
    }


    // Compute Reward
    arma::vec R = (currentState.rows(1,config->Nr)-config->TLB)%(currentState.rows(1,config->Nr)-config->TUB);
    arma::vec maxR = arma::max(R,arma::zeros(config->Nr,1));
    reward[0] = -arma::sum(maxR);


    // Assign Next State
    nextState = currentState;
}


void MultiHeat::getInitialState(DenseState& state)
{
    currentState.setAbsorbing(false);

    // Extract Initial State
    currentState(mode) = 0;
    currentState.rows(1,config->Nr) = (26-13.5)*randu<vec>(config->Nr)+13.5;

    state = currentState;
}

void MultiHeat::computeTransitionMatrix()
{
    Xi = config->A;
    Xi.diag() -= config->B + sum(config->A,1);
    Gamma = config->B * config->Ta;
}

}
