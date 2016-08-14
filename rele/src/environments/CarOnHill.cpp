/*
 * rele,
 *
 *
 * Copyright (C) 2016 Davide Tateo
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

#include "rele/environments/CarOnHill.h"

#include "rele/utils/RandomGenerator.h"

using namespace std;
using namespace boost::numeric::odeint;

namespace ReLe
{

CarOnHillSettings::CarOnHillSettings()
{
    CarOnHillSettings::defaultSettings(*this);
}

void CarOnHillSettings::defaultSettings(CarOnHillSettings& settings)
{
    //Environment Parameters
    settings.gamma = 0.95;
    settings.stateDimensionality = 2;
    settings.actionDimensionality = 1;
    settings.rewardDimensionality = 1;
    settings.statesNumber = 0;
    settings.actionsNumber = 2;
    settings.isFiniteHorizon = true;
    settings.isAverageReward = false;
    settings.isEpisodic = true;
    settings.horizon = 300;

    //CarOnHillway Parameters
    settings.m = 1;

    settings.dt = 0.1;
}

CarOnHillSettings::~CarOnHillSettings()
{

}

void CarOnHillSettings::WriteToStream(std::ostream& out) const
{
    //TODO [SERIALIZATION] implement
}

void CarOnHillSettings::ReadFromStream(std::istream& in)
{
    //TODO [SERIALIZATION] implement
}

CarOnHill::CarOnHillOde::CarOnHillOde(CarOnHillSettings& config) :
    m(config.m), action(0)
{
}

void CarOnHill::CarOnHillOde::operator()(const state_type& x, state_type& dx,
        const double /* t */)
{
    // Status and actions
    const double u = action;
    const double p = x[position];
    const double v = x[velocity];

    // Parameters
    double diffHill;
    double diff2Hill;
    if(p < 0)
    {
        diffHill = 2 * p + 1;
        diff2Hill = 2;
    }
    else
    {
        diffHill = 1 / pow(1 + 5 * x[position] * x[position], 1.5);
        diff2Hill = (-15 * p) /
                    pow(1 + 5 * x[position] * x[position], 2.5);
    }

    // Dynamics
    // Velocity
    const double dPosition = v;

    // Acceleration
    const double dVelocity = u / (m * (1 + diffHill * diffHill)) -
                             (g * diffHill) / (1 + diffHill * diffHill) -
                             (v * v * diffHill * diff2Hill) / (1 + diffHill * diffHill);

    dx.resize(2);
    dx[0] = dPosition;
    dx[1] = dVelocity;
}

///////////////////////////////////////////////////////////////////////////////////////
/// CARONHILL ENVIRONMENT
///////////////////////////////////////////////////////////////////////////////////////

CarOnHill::CarOnHill()
    : DenseMDP(new CarOnHillSettings()),
      cleanConfig(true),
      carOnHillOde(static_cast<CarOnHillSettings&>(getWritableSettings())),
      controlled_stepper(make_controlled< error_stepper_type >(1.0e-6, 1.0e-6))
{
    carOnHillConfig = static_cast<CarOnHillSettings*>(settings);
    currentState.set_size(carOnHillConfig->stateDimensionality);
}

CarOnHill::CarOnHill(CarOnHillSettings& config)
    : DenseMDP(&config),
      cleanConfig(false),
      carOnHillConfig(&config),
      carOnHillOde(*carOnHillConfig),
      controlled_stepper(make_controlled< error_stepper_type >(1.0e-6, 1.0e-6))
{
    currentState.set_size(this->getSettings().stateDimensionality);
}

void CarOnHill::step(const FiniteAction& action,
                     DenseState& nextState, Reward& reward)
{
    //ODEINT (BOOST 1.53+)
    carOnHillOde.action = (action.getActionN() == 0? -4 : 4);
    double t0 = 0;
    double t1 = carOnHillConfig->dt;
    integrate_adaptive(controlled_stepper,
                       carOnHillOde,
                       currentState,
                       t0,
                       t1,
                       t1 / 1000);

    // Compute reward
    if(currentState[position] < -1 || abs(currentState[velocity]) > 3)
    {
        currentState.setAbsorbing();
        reward[0] = -1;
    }
    else if(currentState[position] > 1 && abs(currentState[velocity]) <= 3)
    {
        currentState.setAbsorbing();
        reward[0] = 1;
    }
    else
        reward[0] = 0;

    nextState = currentState;
}

void CarOnHill::getInitialState(DenseState& state)
{
    currentState.setAbsorbing(false);
    currentState[position] = -0.5;
    currentState[velocity] = 0;
    state = currentState;
}

}
