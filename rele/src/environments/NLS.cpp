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

#include "rele/environments/NLS.h"

#include "rele/utils/RandomGenerator.h"
#include <cassert>

#define NLS_NOISE_MEAN      0.0
#define NLS_NOISE_STD       0.02
#define NLS_POS0_NOISE_MEAN 1.0
#define NLS_POS0_NOISE_STD  0.001
#define NLS_REW_AREA        0.1

using namespace std;

namespace ReLe
{

NLSSettings::NLSSettings()
{
    NLSSettings::defaultSettings(*this);
}

void NLSSettings::defaultSettings(NLSSettings& settings)
{
    //Environment Parameters
    settings.gamma = 0.95;
    settings.stateDimensionality = 2;
    settings.actionDimensionality = 1;
    settings.rewardDimensionality = 1;
    settings.statesNumber = -1;
    settings.actionsNumber = -1;
    settings.isFiniteHorizon = false;
    settings.isAverageReward = false;
    settings.isEpisodic = false;
    settings.horizon = 80;

    //NLS parameters
    settings.noise_mean = NLS_NOISE_MEAN;
    settings.noise_std  = NLS_NOISE_STD;
    settings.pos0_mean  = NLS_POS0_NOISE_MEAN;
    settings.pos0_std   = NLS_POS0_NOISE_STD;
    settings.reward_reg = NLS_REW_AREA;
}

void NLSSettings::WriteToStream(ostream &out) const
{
    EnvironmentSettings::writeToStream(out);
    out << this->noise_mean << std::endl;
    out << this->noise_std << std::endl;
    out << this->pos0_mean << std::endl;
    out << this->pos0_std << std::endl;
    out << this->reward_reg << std::endl;
}

void NLSSettings::ReadFromStream(istream &in)
{
    EnvironmentSettings::readFromStream(in);
    in >> this->noise_mean;
    in >> this->noise_std;
    in >> this->pos0_mean;
    in >> this->pos0_std;
    in >> this->reward_reg;
}

NLSSettings::~NLSSettings()
{

}


///////////////////////////////////////////////////////////////////////////////////////
/// NLS ENVIRONMENTS
///////////////////////////////////////////////////////////////////////////////////////

NLS::NLS()
    : ContinuousMDP(new NLSSettings()), cleanConfig(true), config(static_cast<NLSSettings*>(settings))
{
    currentState.set_size(this->getSettings().stateDimensionality);
}

NLS::NLS(NLSSettings& config)
    : ContinuousMDP(&config), cleanConfig(true), config(&config)
{
    currentState.set_size(this->getSettings().stateDimensionality);
}

void NLS::step(const DenseAction& action, DenseState& nextState, Reward& reward)
{
    const NLSSettings& nlsConfig = *config;
    double a           = action[0];
    double model_noise = RandomGenerator::sampleNormal(nlsConfig.noise_mean, nlsConfig.noise_std);

    // model transitions
    //s_2(t+1) = s_2(t) + 1/(1+exp(-u(t))) - 0.5 + noise
    nextState[1] = currentState[1] + 1.0/(1 + exp(-a)) - 0.5 + model_noise;
    //s_1(t+1) = s_1(t) - 0.1 x_2(t+1) + noise
    nextState[0] = currentState[0] - 0.1 * nextState[1] + model_noise;

    double norm2state = sqrt(currentState[0]*currentState[0]+currentState[1]*currentState[1]);
    reward[0] = (norm2state <= nlsConfig.reward_reg) ? 1.0 : 0.0;

    nextState.setAbsorbing(false);
    currentState = nextState;
}

void NLS::getInitialState(DenseState& state)
{
    const NLSSettings& nlsConfig = *config;
    currentState.setAbsorbing(false);
    currentState[0] = RandomGenerator::sampleNormal(nlsConfig.pos0_mean, nlsConfig.pos0_std);
//    currentState[1] = RandomGenerator::sampleNormal(0.0, 1);
    currentState[1] = 0.0;
    state = currentState;
}




}  //end namespace

#undef NLS_NOISE_MEAN
#undef NLS_NOISE_STD
#undef NLS_POS0_NOISE_MEAN
#undef NLS_POS0_NOISE_STD
#undef NLS_REW_AREA
