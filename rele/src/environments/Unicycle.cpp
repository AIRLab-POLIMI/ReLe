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

#include "rele/environments/Unicycle.h"
#include "rele/utils/RandomGenerator.h"
#include "rele/utils/ArmadilloExtensions.h"

using namespace std;
using namespace boost::numeric::odeint;

namespace ReLe
{

//=====================================================================================
// UnicyclePolarSettings SETTINGS
//-------------------------------------------------------------------------------------

UnicyclePolarSettings::UnicyclePolarSettings()
{
    UnicyclePolarSettings::defaultSettings(*this);
}

void UnicyclePolarSettings::defaultSettings(UnicyclePolarSettings& settings)
{
    //Environment Parameters
    settings.gamma = 0.99;
    settings.stateDimensionality = 3;
    settings.actionDimensionality = 2;
    settings.rewardDimensionality = 1;
    settings.statesNumber = 0;
    settings.actionsNumber = 0;
    settings.isFiniteHorizon = false;
    settings.isAverageReward = false;
    settings.isEpisodic = true;
    settings.horizon = 300;

    //Unicycle Parameters
    settings.dt = 0.03; //s
    settings.reward_th = 0.1;
}

UnicyclePolarSettings::~UnicyclePolarSettings()
{

}

void UnicyclePolarSettings::WriteToStream(std::ostream& out) const
{
    //TODO [SERIALIZATION] implement
}

void UnicyclePolarSettings::ReadFromStream(std::istream& in)
{
    //TODO [SERIALIZATION] implement
}

void UnicyclePolar::UnicyclePolarOde::operator ()(const state_type& x, state_type& dx,
        const double /* t */)
{
    //Status and actions
    const double rho   = std::max(x[StateLabel::rho], 1e-6); //avoid numerical instability
    const double gamma = x[StateLabel::gamma];
    // const double delta = std::max(x[StateLabel::delta], 1e-6);

    //dinamics
    const double drho = -v * cos(gamma);

    const double dgamma = sin(gamma) * v / rho - w;

    const double ddelta = sin(gamma) * v / rho;

    dx.set_size(3);
    dx[StateLabel::rho]   = drho;
    dx[StateLabel::gamma] = dgamma;
    dx[StateLabel::delta] = ddelta;

}

//=====================================================================================
// UnicyclePolar ENVIRONMENT
//-------------------------------------------------------------------------------------

UnicyclePolar::UnicyclePolar()
    : ContinuousMDP(new UnicyclePolarSettings()), cleanConfig(true),
      controlled_stepper (make_controlled< error_stepper_type >( 1.0e-6 , 1.0e-6 ))
{
    unicycleConfig = static_cast<UnicyclePolarSettings*>(settings);
    currentState.set_size(unicycleConfig->stateDimensionality);
}

UnicyclePolar::UnicyclePolar(UnicyclePolarSettings& config)
    : ContinuousMDP(&config), cleanConfig(false), unicycleConfig(&config),
      controlled_stepper (make_controlled< error_stepper_type >( 1.0e-6 , 1.0e-6 ))
{
    currentState.set_size(this->getSettings().stateDimensionality);
}

void UnicyclePolar::step(const DenseAction& action, DenseState& nextState, Reward& reward)
{
    double v = action[linearVel];
    double w = action[angularVel];

    //ODEINT (BOOST 1.53+) -- integrate
    unicycleode.v = v;
    unicycleode.w = w;
    double t0 = 0;
    double t1 = unicycleConfig->dt;
    integrate_adaptive(controlled_stepper , unicycleode , currentState, t0 , t1 , t1/100.0);

    // wrap to [-pi,pi]
    currentState[gamma] = wrapToPi(currentState[gamma]);
    currentState[delta] = wrapToPi(currentState[delta]);

    // update next state
    nextState = currentState;

    //compute reward
//    arma::vec& vecState = nextState;
//    double dist = arma::norm(vecState,2);
    double dist = (abs(nextState(0)) + 10 * abs(nextState(1)) + 10 * abs(nextState(2)));
    reward[0] = -dist - 0.1 * w*w - 0.05 * v*v;

    if (dist < unicycleConfig->reward_th)
    {
        nextState.setAbsorbing(true);
    }

}

void UnicyclePolar::getInitialState(DenseState& state)
{
    double goal[] = {0,0,0}; // rad
    double x = RandomGenerator::sampleUniform(-4,4);
    double y = RandomGenerator::sampleUniform(-4,4);
    double theta = RandomGenerator::sampleUniform(-M_PI,M_PI);

//    x = 4;
//    y = 4;
//    theta = M_PI;

    arma::mat Tr(3,3);
    Tr <<  cos(goal[2]) << sin(goal[2]) << 0.0 << arma::endr
       << -sin(goal[2]) << cos(goal[2]) << 0.0 << arma::endr
       << 0.0 << 0.0 << 1.0;

    arma::vec e(3);
    e(0) = x-goal[0];
    e(1) = y-goal[1];
    e(2) = theta-goal[2];

    e = Tr * e;

    // set initial state
    currentState[rho]   = sqrt(e(0) * e(0) + e(1) * e(1));
    currentState[gamma] = atan2(e(1), e(0)) - e(2) + M_PI;
    currentState[delta] = currentState[gamma] + e(2);

    // wrap to [-pi,pi]
    currentState[gamma] = wrapToPi(currentState[gamma]);
    currentState[delta] = wrapToPi(currentState[delta]);

    // set not absorbing
    currentState.setAbsorbing(false);
    state = currentState;
}

//=====================================================================================
// UnicycleControlLaw POLICY
//-------------------------------------------------------------------------------------
arma::vec UnicycleControlLaw::operator()(const arma::vec &state)
{
    arma::vec action(2);
    double k1 = params(0), k2 = params(1), k3 = params(2);

    double rho   = state(UnicyclePolar::StateLabel::rho);
    double gamma = state(UnicyclePolar::StateLabel::gamma);
    double delta = state(UnicyclePolar::StateLabel::delta);

    action(UnicyclePolar::ActionLabel::linearVel)  = k1 * rho * cos(gamma);
    action(UnicyclePolar::ActionLabel::angularVel) = k2 * gamma + k1 * sin(gamma) * cos(gamma) * (gamma + k3 * delta) / gamma;

    return action;
}

double UnicycleControlLaw::operator()(const arma::vec& state, const arma::vec& action)
{
    arma::vec realAction = (*this)(state);
    if ((action(0) == realAction(0)) && (action(1) == realAction(1)))
        return 1;
    return 0;
}

Policy<DenseAction, DenseState>* UnicycleControlLaw::clone()
{
    return new UnicycleControlLaw(this->getParameters());
}

arma::vec UnicycleControlLaw::getParameters() const
{
    return params;
}

const unsigned int UnicycleControlLaw::getParametersSize() const
{
    return params.n_elem;
}

void UnicycleControlLaw::setParameters(const arma::vec& w)
{
    assert(params.n_elem == w.n_elem);
    params = w;
}

}
