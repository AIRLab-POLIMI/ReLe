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

#include "rele/environments/Dam.h"

#include "rele/utils/RandomGenerator.h"
#include <cassert>

using namespace std;

namespace ReLe
{

DamSettings::DamSettings()
{
    DamSettings::defaultSettings(*this);
}

void DamSettings::defaultSettings(DamSettings& settings)
{
    //Environment Parameters
    settings.gamma = 1.0;
    settings.stateDimensionality = 1;
    settings.actionDimensionality = 1;
    settings.rewardDimensionality = 4;
    settings.statesNumber = -1;
    settings.actionsNumber = -1;
    settings.isFiniteHorizon = false;
    settings.isAverageReward = true;
    settings.isEpisodic = false;
    settings.horizon = 100;

    //Dam Parameters
    settings.S = 1.0;                      // reservoir surface
    settings.W_IRR = 50.0;                 // water demand
    settings.H_FLO_U = 50.0;               // flooding threshold
    settings.S_MIN_REL = 100.0;
    settings.DAM_INFLOW_MEAN = 40;
    settings.DAM_INFLOW_STD  = 10;
    settings.Q_MEF = 0.0;
    settings.GAMMA_H2O = 1000.0;
    settings.W_HYD = 4.36;                 //  hydroelectric demand
    settings.Q_FLO_D = 30.0;
    settings.ETA = 1.0;
    settings.G = 9.81;

    settings.max_obj.set_size(4);
    settings.max_obj(0) = 20;
    settings.max_obj(1) = 20;
    settings.max_obj(2) = 2;
    settings.max_obj(3) = 1;

    settings.penalize = false;
    settings.initial_state_type = initType::RANDOM;
}

void DamSettings::WriteToStream(ostream &out) const
{
    EnvironmentSettings::writeToStream(out);
    out << this->S << std::endl;
    out << this->W_IRR << std::endl;
    out << this->H_FLO_U << std::endl;
    out << this->S_MIN_REL << std::endl;
    out << this->DAM_INFLOW_MEAN << std::endl;
    out << this->DAM_INFLOW_STD << std::endl;
    out << this->Q_MEF << std::endl;
    out << this->GAMMA_H2O << std::endl;
    out << this->W_HYD << std::endl;
    out << this->Q_FLO_D << std::endl;
    out << this->ETA << std::endl;
    out << this->G << std::endl;
    out << this->max_obj.size() << std::endl;
    for (int i = 0, ie = this->max_obj.size(); i < ie; ++i)
    {
        out << this->max_obj[i] << "\t";
    }
    out << std::endl;
}

void DamSettings::ReadFromStream(istream &in)
{
    EnvironmentSettings::readFromStream(in);
    in >> this->S >> this->W_IRR;
    in >> this->H_FLO_U >> this->S_MIN_REL;
    in >> this->DAM_INFLOW_MEAN >> this->DAM_INFLOW_STD;
    in >> this->Q_MEF >> this->GAMMA_H2O;
    in >> this->W_HYD >> this->Q_FLO_D;
    in >> this->ETA >> this->G;
    int dim, y;
    in >> dim;
    max_obj.set_size(dim);
    for (int i = 0; i < dim; ++i)
    {
        in >> y;
        this->max_obj(i) = y;
    }
}


///////////////////////////////////////////////////////////////////////////////////////
/// DAM ENVIRONMENTS
///////////////////////////////////////////////////////////////////////////////////////

Dam::Dam()
    : ContinuousMDP(new DamSettings()), cleanConfig(true), config(static_cast<DamSettings*>(settings)), nbSteps(0)
{
    currentState.set_size(this->getSettings().stateDimensionality);
}

Dam::Dam(DamSettings& config)
    : ContinuousMDP(&config), cleanConfig(false), config(&config), nbSteps(0)
{
    currentState.set_size(this->getSettings().stateDimensionality);
}

void Dam::step(const DenseAction& action, DenseState& nextState, Reward& reward)
{
    const DamSettings& damConfig = *config;
    // bound the action
    double min_action = std::max(currentState[0] - damConfig.S_MIN_REL, 0.0);
    double max_action = currentState[0];

    double penalty = 0.0;

    double curr_action = action[0];

    if ((min_action > curr_action) || (max_action < curr_action))
    {

        penalty = -std::max(curr_action - max_action, min_action - curr_action);
        curr_action  = std::max(min_action, std::min(max_action, curr_action));

    }

    if (damConfig.penalize == false)
    {
        penalty = 0.0;
    }

    // transition dynamic
    double inflow = RandomGenerator::sampleNormal(damConfig.DAM_INFLOW_MEAN, damConfig.DAM_INFLOW_STD);
    //    std::cout << "inflow: " << inflow << std::endl;
    nextState[0]  = currentState[0] + inflow - curr_action;

    // cost due to the excess level w.r.t. a flooding threshold (upstream)
    reward[0] = -std::max(nextState[0]/damConfig.S - damConfig.H_FLO_U, 0.0) + penalty;

    if (damConfig.rewardDimensionality >= 2)
    {
        // deficit in the water supply w.r.t. the water demand
        reward[1] = -std::max(damConfig.W_IRR - curr_action, 0.0) + penalty;
    }

    double q = 0.0;
    if (curr_action > damConfig.Q_MEF)
    {
        q = curr_action - damConfig.Q_MEF;
        //        std::cout << "q: " << q << std::endl;
    }
    double p_hyd = damConfig.ETA * damConfig.G * damConfig.GAMMA_H2O *
                   (nextState[0]/damConfig.S) * (q / (3.6e6));
    //    std::cout << "damConfig.ETA " << damConfig.ETA << "\n";
    //    std::cout << "damConfig.G " << damConfig.G << "\n";
    //    std::cout << "damConfig.GAMMA_H20 " << damConfig.GAMMA_H2O << "\n";
    //    std::cout << "damConfig.S " << damConfig.S << "\n";
    //    std::cout << "nextstate " << nextstate[0] << "\n";
    //    std::cout << "p_hyd: " << p_hyd << std::endl;

    if (damConfig.rewardDimensionality >= 3)
    {
        // deficit in the hydroelectric supply w.r.t to hydroelectric demand
        reward[2] = -std::max(damConfig.W_HYD - p_hyd, 0.0) + penalty;
    }

    if (damConfig.rewardDimensionality >= 4)
    {
        // cost due to the excess level w.r.t. a flooding threshold (downstream)
        reward[3] = -std::max(curr_action - damConfig.Q_FLO_D, 0.0) + penalty;
    }

    nextState.setAbsorbing(false);

    for (unsigned int i = 0, ie = damConfig.rewardDimensionality; i < ie; ++i)
    {
        reward[i] /= damConfig.max_obj[i];
        if (damConfig.isAverageReward)
            reward[i] /= getSettings().horizon;
    }
//    std::cout << currentState << " | " << action << " | " <<  nextState << std::endl;

    currentState = nextState;
    ++nbSteps;
}

void Dam::getInitialState(DenseState& state)
{
    const DamSettings& damConfig = *config;

    if (damConfig.isAverageReward && nbSteps != 0)
    {
        assert(nbSteps == getSettings().horizon);
    }

    state.setAbsorbing(false);

    if (damConfig.initial_state_type == DamSettings::initType::RANDOM_DISCRETE)
    {
//        std::cout << "RANDOM_DISCRETE" << std::endl;
        // initial states
        double s_init[] = {9.6855361e+01, 5.8046026e+01,
                           1.1615767e+02, 2.0164311e+01,
                           7.9191000e+01, 1.4013098e+02,
                           1.3101816e+02, 4.4351321e+01,
                           1.3185943e+01, 7.3508622e+01
                          };
        int idx   = RandomGenerator::sampleUniformInt(0,10);
        currentState[0] = s_init[idx]; //keep info about the current state
    }
    else if (damConfig.initial_state_type == DamSettings::initType::RANDOM)
    {
//        std::cout << "RANDOM" << std::endl;
        currentState[0] = RandomGenerator::sampleUniformInt(0,160);
    }
    else
    {
        std::cerr << "getInitialState: something wrong!" << std::endl;
        abort();
    }
    state = currentState;
    nbSteps = 0;
}

void Dam::setCurrentState(const DenseState &state)
{
    currentState[0] = state[0];
}




}  //end namespace
