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

#include "UnderwaterVehicle.h"
#include "RandomGenerator.h"
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
    settings.continuosStateDim = 1;
    settings.continuosActionDim = -1;
    settings.rewardDim = 4;
    settings.finiteStateDim = -1;
    settings.finiteActionDim = 5;
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
    settings.actionList = {settings.thrustRange.Lo(), settings.thrustRange.Lo()/2.0,
                           0.0, settings.thrustRange.Hi()/2.0, settings.thrustRange.Hi()
                          };
}

void UWVSettings::WriteToStream(ostream &out) const
{
    //TODO
}

void UWVSettings::ReadFromStream(istream &in)
{
    //TODO
}


///////////////////////////////////////////////////////////////////////////////////////
/// UNDERWATER VEHICLE ENVIRONMENTS
///////////////////////////////////////////////////////////////////////////////////////

UnderwaterVehicle::UnderwaterVehicle()
    : uwvConfig(),
      DenseMDP(uwvConfig.continuosStateDim,uwvConfig.finiteActionDim,uwvConfig.rewardDim,
               uwvConfig.isFiniteHorizon, uwvConfig.isEpisodic, uwvConfig.gamma, uwvConfig.horizon),
      cState(uwvConfig.continuosStateDim), uwvode(),
      controlled_stepper (make_controlled< error_stepper_type >( 1.0e-6 , 1.0e-6 ))
{
}

UnderwaterVehicle::UnderwaterVehicle(UWVSettings &config)
    : uwvConfig(config),
      DenseMDP(uwvConfig.continuosStateDim,uwvConfig.finiteActionDim,uwvConfig.rewardDim,
               uwvConfig.isFiniteHorizon, uwvConfig.isEpisodic, uwvConfig.gamma, uwvConfig.horizon),
      cState(uwvConfig.continuosStateDim), uwvode(),
      controlled_stepper (make_controlled< error_stepper_type >( 1.0e-6 , 1.0e-6 ))
{
}

void UnderwaterVehicle::step(const FiniteAction& action, DenseState& nextState, Reward& reward)
{
    double u = uwvConfig.actionList[action.getActionN()];

    //ODEINT (BOOST 1.53+)
    uwvode.action = u;
    double t0 = 0;
    double t1 = uwvConfig.dt;
    integrate_adaptive( controlled_stepper , uwvode , cState, t0 , t1 , t1/100.0);

    ////Runge-Kutta 4/5
    //double** odeTime;
    //double** odeState;
    //unsigned int ndata;
    //ode45(UnderwaterVehicle::ode, &u, 0, uwvConfig.dt,
    //      cState.memptr(), cState.n_elem,
    //      odeTime, odeState, ndata);
    //nextState[0] = odeState[0][ndata-1];
    //nextState.setAbsorbing(false);
    //Cstate = nextState;
    ////free variables
    //free(odeState);
    //free(odeTime);

    nextState = cState;

    reward[0] = fabs(uwvConfig.setPoint - nextState[0]) < uwvConfig.mu ? 0.0f : -uwvConfig.C;

}

void UnderwaterVehicle::getInitialState(DenseState &state)
{
    state[0] = RandomGenerator::sampleUniform(uwvConfig.velocityRange.Lo(),uwvConfig.velocityRange.Hi());
    state.setAbsorbing(false);
    cState[0] = state[0]; //keep info about the current state
}




}  //end namespace
