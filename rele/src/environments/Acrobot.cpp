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

#include "rele/environments/Acrobot.h"
#include "rele/utils/ModularRange.h"
#include "rele/utils/RandomGenerator.h"

using namespace std;
using namespace boost::numeric::odeint;

namespace ReLe
{

AcrobotSettings::AcrobotSettings()
{
    AcrobotSettings::defaultSettings(*this);
}

void AcrobotSettings::defaultSettings(AcrobotSettings& settings)
{
    //Environment Parameters
    settings.gamma = 0.95;
    settings.stateDimensionality = 4;
    settings.actionDimensionality = 1;
    settings.rewardDimensionality = 1;
    settings.statesNumber = 0;
    settings.actionsNumber = 2;
    settings.isFiniteHorizon = true;
    settings.isAverageReward = false;
    settings.isEpisodic = true;
    settings.horizon = 100;

    //Acrobot Parameters
    settings.M1 = settings.M2 = 1;
    settings.L1 = settings.L2 = 1;
    settings.mu1 = settings.mu2 = 0.01;

    settings.dt = 0.1;
}

AcrobotSettings::~AcrobotSettings()
{

}

void AcrobotSettings::WriteToStream(std::ostream& out) const
{
    //TODO [SERIALIZATION] implement
}

void AcrobotSettings::ReadFromStream(std::istream& in)
{
    //TODO [SERIALIZATION] implement
}

Acrobot::AcrobotOde::AcrobotOde(AcrobotSettings& config) :
    M1(config.M1),
    M2(config.M2),
    L1(config.L1),
    L2(config.L2),
    mu1(mu1),
    mu2(mu2),
    action(0)
{
}

void Acrobot::AcrobotOde::operator()(const state_type& x, state_type& dx,
                                     const double /* t */)
{
    // Status and actions
    const double u = action;
    const double theta1 = x[theta1idx];
    const double theta2 = x[theta2idx];
    const double dTheta1 = x[dTheta1idx];
    const double dTheta2 = x[dTheta2idx];

    // Parameters
    const double d11 = M1 * L1 * L1 + M2 * (L1 * L1 + L2 * L2 + 2 * L1 * L2 * cos(theta2));
    const double d22 = M2 * L2 * L2;
    const double d12 = M2 * (L2 * L2 + L1 * L2 * cos(theta2));
    const double c1 = -M2 * L1 * L2 * dTheta2 * (2 * dTheta1 + dTheta2 * sin(theta2));
    const double c2 = M2 * L1 * L2 * dTheta1 * dTheta1 * sin(theta2);
    const double phi1 = (M1 * L1 + M2 * L1) * g * sin(theta1) + M2 * L2 * g * sin(theta1 + theta2);
    const double phi2 = M2 * L2 * g * sin(theta1 + theta2);

    // Dynamics
    // Velocity
    const double diffTheta1 = dTheta1;
    const double diffTheta2 = dTheta2;

    // Acceleration
    const double d12d22 = d12 / d22;
    const double diffDiffTheta1 = (-mu1 * dTheta1 - d12d22 * u + d12d22 * mu2 * dTheta2 +
                                   d12d22 * c2 + d12d22 * phi2 - c1 - phi1) / (d11 - (d12d22 * d12));
    const double diffDiffTheta2 = (u - mu2 * dTheta2 - d12 * diffDiffTheta1 - c2 - phi2) / d22;

    dx.resize(4);
    dx[0] = diffTheta1;
    dx[1] = diffTheta2;
    dx[2] = diffDiffTheta1;
    dx[3] = diffDiffTheta2;
}

///////////////////////////////////////////////////////////////////////////////////////
/// ACROBOT ENVIRONMENT
///////////////////////////////////////////////////////////////////////////////////////

Acrobot::Acrobot()
    : DenseMDP(new AcrobotSettings()),
      cleanConfig(true),
      acrobotOde(static_cast<AcrobotSettings&>(getWritableSettings())),
      controlled_stepper(make_controlled< error_stepper_type >(1.0e-6, 1.0e-6))
{
    acrobotConfig = static_cast<AcrobotSettings*>(settings);
    currentState.set_size(acrobotConfig->stateDimensionality);
}

Acrobot::Acrobot(AcrobotSettings& config)
    : DenseMDP(&config),
      cleanConfig(false),
      acrobotConfig(&config),
      acrobotOde(*acrobotConfig),
      controlled_stepper(make_controlled< error_stepper_type >(1.0e-6, 1.0e-6))
{
    currentState.set_size(this->getSettings().stateDimensionality);
}

void Acrobot::step(const FiniteAction& action,
                   DenseState& nextState, Reward& reward)
{
    //ODEINT (BOOST 1.53+)
    acrobotOde.action = (action.getActionN() == 0? -5 : 5);
    double t0 = 0;
    double t1 = acrobotConfig->dt;
    integrate_adaptive(controlled_stepper,
                       acrobotOde,
                       currentState,
                       t0,
                       t1,
                       0.001);

    // Compute reward
    int k = round((currentState[theta1idx] - M_PI) / (2 * M_PI));
#ifndef ARMA_USE_CXX11
    arma::vec x(4);
    x(0) = currentState[theta1idx];
    x(1) = currentState[theta2idx];
    x(2) = currentState[dTheta1idx];
    x(3) = currentState[dTheta2idx];
    arma::vec o = arma::zeros<arma::vec>(4);
    o(0) = 2 * k * M_PI + M_PI;
#else
    arma::vec x = {currentState[theta1idx],
                   currentState[theta2idx],
                   currentState[dTheta1idx],
                   currentState[dTheta2idx]
                  };
    arma::vec o = {2 * k * M_PI + M_PI, 0, 0, 0};
#endif
    arma::vec diffVector = x - o;
    double d = arma::norm(diffVector);
    if(d < 1)
    {
        currentState.setAbsorbing();
        reward[0] = 1 - d;
    }
    else
        reward[0] = 0;

    currentState[theta1idx] = RangePi::wrap(currentState[theta1idx]);
    currentState[theta2idx] = RangePi::wrap(currentState[theta2idx]);
    nextState = currentState;
}

void Acrobot::getInitialState(DenseState& state)
{
    currentState.setAbsorbing(false);
    currentState[theta1idx] = rg.sampleUniform(-M_PI + 1, M_PI - 1);
    currentState[theta2idx] = 0;
    currentState[dTheta1idx] = 0;
    currentState[dTheta2idx] = 0;
    state = currentState;
}

}
