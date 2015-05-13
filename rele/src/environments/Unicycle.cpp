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

#include "Unicycle.h"
#include "RandomGenerator.h"

using namespace std;
using namespace boost::numeric::odeint;

namespace ReLe
{

UnicyclePolarSettings::UnicyclePolarSettings()
{
    UnicyclePolarSettings::defaultSettings(*this);
}

void UnicyclePolarSettings::defaultSettings(UnicyclePolarSettings& settings)
{
    //Environment Parameters
    settings.gamma = 0.99;
    settings.continuosStateDim = 3;
    settings.continuosActionDim = 2;
    settings.rewardDim = 1;
    settings.finiteStateDim = 0;
    settings.finiteActionDim = 0;
    settings.isFiniteHorizon = false;
    settings.isAverageReward = false;
    settings.isEpisodic = true;
    settings.horizon = 300;

    //UWV Parameters
    settings.dt = 0.03; //s
}

UnicyclePolarSettings::~UnicyclePolarSettings()
{

}

void UnicyclePolarSettings::WriteToStream(std::ostream& out) const
{
    //TODO implement
}

void UnicyclePolarSettings::ReadFromStream(std::istream& in)
{
    //TODO implement
}

void UnicyclePolar::UnicyclePolarOde::operator ()(const state_type& x, state_type& dx,
        const double /* t */)
{
    //Status and actions
    const double rho   = x[0];
    const double gamma = x[1];
    const double delta = x[2];

    //dinamics
    const double drho = -v * cos(gamma);

    const double dgamma = sin(gamma) * v / rho - w;

    const double ddelta = sin(gamma) * v / rho;

    dx.resize(3);
    dx[0] = drho;
    dx[1] = dgamma;
    dx[2] = ddelta;

}

///////////////////////////////////////////////////////////////////////////////////////
/// UnicyclePolar ENVIRONMENT
///////////////////////////////////////////////////////////////////////////////////////

UnicyclePolar::UnicyclePolar()
    : ContinuousMDP(new UnicyclePolarSettings()), cleanConfig(true),
      controlled_stepper (make_controlled< error_stepper_type >( 1.0e-6 , 1.0e-6 ))
{
    unicycleConfig = static_cast<UnicyclePolarSettings*>(settings);
    currentState.set_size(unicycleConfig->continuosStateDim);
}

UnicyclePolar::UnicyclePolar(UnicyclePolarSettings& config)
    : ContinuousMDP(&config), cleanConfig(false), unicycleConfig(&config),
      controlled_stepper (make_controlled< error_stepper_type >( 1.0e-6 , 1.0e-6 ))
{
    currentState.set_size(this->getSettings().continuosStateDim);
}

void UnicyclePolar::step(const DenseAction& action, DenseState& nextState, Reward& reward)
{
    double v = action[0];
    double w = action[1];

    //ODEINT (BOOST 1.53+)
    unicycleode.v = v;
    unicycleode.w = w;
    double t0 = 0;
    double t1 = unicycleConfig->dt;
    integrate_adaptive(controlled_stepper , unicycleode , currentState, t0 , t1 , t1/1000.0);

    nextState = currentState;

    //compute reward
    arma::vec& vecState = nextState;
    double dist = arma::norm(vecState,2);
    reward[0] = -dist;

}

void UnicyclePolar::getInitialState(DenseState &state)
{
    double goal[] = {0,0,0}; // rad
    double x = RandomGenerator::sampleUniform(-4,4);
    double y = RandomGenerator::sampleUniform(-4,4);
    double theta = RandomGenerator::sampleUniform(-M_PI,M_PI);

    arma::mat Tr(3,3);
    Tr <<  cos(goal[2]) << sin(goal[2]) << 0.0 << arma::endr
       << -sin(goal[2]) << cos(goal[2]) << 0.0 << arma::endr
       << 0.0 << 0.0 << 1.0;

    arma::vec e(3);
    e(0) = x-goal[0];
    e(1) = y-goal[1];
    e(2) = theta-goal[2];

    e = Tr * e;

    currentState[0] = sqrt(e(0) * e(0) + e(1) * e(1));
    currentState[1] = atan2(e(1),e(0)) - e(2) + M_PI;
    currentState[2] = currentState[1] + e(2);
    currentState.setAbsorbing(false);
    state = currentState;
}



}
