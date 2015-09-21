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

#include "Pursuer.h"

#include "ModularRange.h"
#include "RandomGenerator.h"

using namespace arma;
using namespace std;

namespace ReLe
{

Pursuer::Pursuer() :
    ContinuousMDP(STATESIZE, 3, 1, false, true, 0.99, 1000), dt(0.01),
    maxOmega(-M_PI, M_PI), maxV(0, 1),
    maxOmegar(-(M_PI-M_PI/12), M_PI-M_PI/12), maxVr(0, 1),
    limitX(-10, 10), limitY(-10, 10), predictor(dt, limitX, limitY)
{

}

void Pursuer::step(const DenseAction& action, DenseState& nextState,
                   Reward& reward)
{
    //action threshold
    double v = maxV.bound(action[0]);
    double omega = maxOmega.bound(action[1]);

    //Compute pursuer control using chicken pose prediction
    double omegar;
    double vr;
    computePursuerControl(vr, omegar);

    //Update pursuer state
    double xrabs, yrabs;
    updatePursuerPose(vr, omegar, xrabs, yrabs);

    //update chased position
    updateChasedPose(v, omega);

    //update pursuer relative position
    currentState[xp] = xrabs - currentState[x];
    currentState[yp] = yrabs - currentState[y];

    if (captured())
    {
        reward[0] = -100;
        currentState.setAbsorbing(true);
    }
    else
    {
        reward[0] = 1;
    }

    nextState = currentState;
}

void Pursuer::getInitialState(DenseState& state)
{
    //chased state
    currentState[x] = RandomGenerator::sampleUniform(limitX.lo(), limitX.hi());
    currentState[y] = RandomGenerator::sampleUniform(limitY.lo(), limitY.hi());
    currentState[theta] = RandomGenerator::sampleUniform(-M_PI, M_PI);

    //pursuer state
    do
    {
        currentState[xp] = RandomGenerator::sampleNormal(0.0, 1.0);
        currentState[yp] = RandomGenerator::sampleNormal(0.0, 1.0);
    }
    while(!feasibleState());

    currentState[thetap] = RandomGenerator::sampleUniform(-M_PI, M_PI);

    //reset predictor state
    predictor.reset();

    currentState.setAbsorbing(false);

    state = currentState;
}

void Pursuer::computePursuerControl(double& vr, double& omegar)
{
    //Predict chicken position
    double xhat, yhat, thetaDirhat;
    predictor.predict(currentState, xhat, yhat, thetaDirhat);

    //Compute rocky control signals
    double deltaTheta = RangePi::wrap(thetaDirhat - currentState[thetap]);
    double omegarOpt = deltaTheta / dt;

    omegar = maxOmegar.bound(omegarOpt);

    if (abs(deltaTheta) > M_PI / 2)
    {
        vr = 0;
    }
    else if (abs(deltaTheta) > M_PI / 4)
    {
        vr = maxVr.hi() / 2;
    }
    else
    {
        vr = maxVr.hi();
    }
}

void Pursuer::updatePursuerPose(double vr, double omegar, double& xrabs,
                                double& yrabs)
{
    vec2 chickenPosition = currentState.rows(span(x, y));
    vec2 rockyRelPosition = currentState.rows(span(xp, yp));

    double thetarM = (2 * currentState[thetap] + omegar * dt) / 2;
    currentState[thetap] = RangePi::wrap(
                               currentState[thetap] + omegar * dt);
    xrabs = chickenPosition[0] + rockyRelPosition[0] + vr * cos(thetarM) * dt;
    yrabs = chickenPosition[1] + rockyRelPosition[1] + vr * sin(thetarM) * dt;

    //Anelastic walls
    xrabs = limitX.bound(xrabs);
    yrabs = limitY.bound(yrabs);
}

void Pursuer::updateChasedPose(double v, double omega)
{
    double thetaM = (2 * currentState[theta] + omega * dt) / 2;
    currentState[x] += v * cos(thetaM) * dt;
    currentState[y] += v * sin(thetaM) * dt;

    //Anelastic walls
    currentState[x] = limitX.bound(currentState[x]);
    currentState[y] = limitY.bound(currentState[y]);

    currentState[theta] = RangePi::wrap(
                              currentState[theta] + omega * dt);

    predictor.saveLastValues(thetaM, v);
}


bool Pursuer::feasibleState()
{
    bool cond1 = limitX.contains(currentState[x] + currentState[xp]);
    bool cond2 = cond1 && limitY.contains(currentState[y] + currentState[yp]);

    return cond2 && !captured();
}

bool Pursuer::captured()
{
    vec2 pursuerRelPosition = currentState.rows(span(xp, yp));
    return norm(pursuerRelPosition) < 0.05;
}

Pursuer::Predictor::Predictor(double dt, Range limitX, Range limitY) :
    dt(dt), limitX(limitX), limitY(limitY)
{
    reset();
}

void Pursuer::Predictor::reset()
{
    thetaM = 0;
    v = 0;
}

void Pursuer::Predictor::saveLastValues(double thetaM, double v)
{
    this->thetaM = thetaM;
    this->v = v;
}

void Pursuer::Predictor::predict(const DenseState& state, double& xhat, double& yhat, double& thetaDirhat)
{
    xhat = state[x] + v * cos(thetaM) * dt;
    yhat = state[y] + v * sin(thetaM) * dt;

    //Anelastic walls
    xhat = limitX.bound(xhat);
    yhat = limitY.bound(yhat);

    thetaDirhat = RangePi::wrap(atan2(yhat - (state[y] + state[yp]), xhat - (state[x] + state[xp])));
}

}
