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

#include "rele/environments/UnderwaterVehicle.h"

#include "rele/utils/RandomGenerator.h"
#include <cassert>

using namespace std;
using namespace boost::numeric::odeint;


namespace ReLe
{

UWVSettings::UWVSettings()
{
    UWVSettings::defaultSettings(*this);
}

void UWVSettings::defaultSettings(UWVSettings& settings)
{
    //Environment Parameters
    settings.gamma = 0.99;
    settings.stateDimensionality = 1;
    settings.actionDimensionality = -1;
    settings.rewardDimensionality = 4;
    settings.statesNumber = -1;
    settings.actionsNumber = 5;
    settings.isFiniteHorizon = false;
    settings.isAverageReward = false;
    settings.isEpisodic = false;
    settings.horizon = 800;

    //UWV Parameters
    settings.thrustRange = Range(-30,30);
    settings.velocityRange = Range(-5,5);
    settings.dt = 0.03; //s
    settings.C = 0.01;
    settings.mu = 0.3; //rad
    settings.setPoint = 4; // m/s
    settings.timeSteps = settings.horizon;
    settings.actionList = {settings.thrustRange.lo(), settings.thrustRange.lo()/2.0,
                           0.0, settings.thrustRange.hi()/2.0, settings.thrustRange.hi()
                          };
}

UWVSettings::~UWVSettings()
{

}

void UWVSettings::WriteToStream(ostream& out) const
{
    //TODO [SERIALIZATION] implement
}

void UWVSettings::ReadFromStream(istream& in)
{
    //TODO [SERIALIZATION] implement
}


///////////////////////////////////////////////////////////////////////////////////////
/// UNDERWATER VEHICLE ENVIRONMENTS
///////////////////////////////////////////////////////////////////////////////////////

UnderwaterVehicle::UnderwaterVehicle()
    : DenseMDP(new UWVSettings()), cleanConfig(true), uwvode(), config(static_cast<UWVSettings*>(settings)),
      controlled_stepper (make_controlled< error_stepper_type >( 1.0e-6 , 1.0e-6 ))
{
    currentState.set_size(this->getSettings().stateDimensionality);
}

UnderwaterVehicle::UnderwaterVehicle(UWVSettings& config)
    : DenseMDP(&config), cleanConfig(false), config(&config), uwvode(),
      controlled_stepper (make_controlled< error_stepper_type >( 1.0e-6 , 1.0e-6 ))
{
    currentState.set_size(this->getSettings().stateDimensionality);
}

void UnderwaterVehicle::step(const FiniteAction& action, DenseState& nextState, Reward& reward)
{
    double u = config->actionList[action.getActionN()];

    //ODEINT (BOOST 1.53+)
    uwvode.action = u;
    double t0 = 0;
    double t1 = config->dt;
    integrate_adaptive( controlled_stepper , uwvode , currentState, t0 , t1 , t1/1000.0);

    nextState = currentState;

    reward[0] = abs(config->setPoint - nextState[0]) < config->mu ? 0.0 : -config->C;

}

void UnderwaterVehicle::getInitialState(DenseState& state)
{
    currentState[0] = RandomGenerator::sampleUniform(config->velocityRange.lo(), config->velocityRange.hi());
    currentState.setAbsorbing(false);
    state = currentState;
}




}  //end namespace
