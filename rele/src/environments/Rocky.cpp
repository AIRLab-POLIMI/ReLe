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

#include "rele/environments/Rocky.h"

#include "rele/utils/ModularRange.h"

using namespace arma;
using namespace std;

namespace ReLe
{

Rocky::Rocky() :
    ContinuousMDP(STATESIZE, 3, 1, false, true, 0.9999), dt(0.01),
    maxOmega(-M_PI, M_PI), maxV(0, 1),
    maxOmegar(-(M_PI-M_PI/12), M_PI-M_PI/12), maxVr(0, 1),
    limitX(-10, 10), limitY(-10, 10),
    maxEnergy(0, 100), predictor(dt, limitX, limitY)
{
    //TODO [MINOR] add parameter in the constructor
    vec2 spot;
    spot[0] = 5;
    spot[1] = 0;
    foodSpots.push_back(spot);
}

void Rocky::step(const DenseAction& action, DenseState& nextState,
                 Reward& reward)
{
    //action threshold
    double v = maxV.bound(action[0]);
    double omega = maxOmega.bound(action[1]);
    bool eat = (action[2] > 0 && v == 0 && omega == 0) ? true : false;

    //Compute rocky control using chicken pose prediction
    double omegar;
    double vr;
    computeRockyControl(vr, omegar);

    //Update rocky state
    double xrabs, yrabs;
    updateRockyPose(vr, omegar, xrabs, yrabs);

    //update chicken position
    updateChickenPose(v, omega);

    //update rocky relative position
    currentState[xr] = xrabs - currentState[x];
    currentState[yr] = yrabs - currentState[y];

    //Compute sensors
    computeSensors(eat);

    //Compute reward
    computeReward(reward);

    nextState = currentState;
}

void Rocky::getInitialState(DenseState& state)
{
    //chicken state
    currentState[x] = 0;
    currentState[y] = 0;
    currentState[theta] = 0;

    //sensors state
    currentState[energy] = 0;
    currentState[food] = 0;

    //rocky state
    currentState[xr] = 1;
    currentState[yr] = 1;
    currentState[thetar] = 0;

    //reset predictor state
    predictor.reset();

    currentState.setAbsorbing(false);

    state = currentState;
}

void Rocky::computeRockyControl(double& vr, double& omegar)
{
    //Predict chicken position
    double xhat, yhat, thetaDirhat;
    predictor.predict(currentState, xhat, yhat, thetaDirhat);

    //Compute rocky control signals
    double deltaTheta = RangePi::wrap(thetaDirhat - currentState[thetar]);
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

void Rocky::updateRockyPose(double vr, double omegar, double& xrabs,
                            double& yrabs)
{
    vec2 chickenPosition = currentState.rows(span(x, y));
    vec2 rockyRelPosition = currentState.rows(span(xr, yr));

    double thetarM = (2 * currentState[thetar] + omegar * dt) / 2;
    currentState[thetar] = RangePi::wrap(
                               currentState[thetar] + omegar * dt);
    xrabs = chickenPosition[0] + rockyRelPosition[0] + vr * cos(thetarM) * dt;
    yrabs = chickenPosition[1] + rockyRelPosition[1] + vr * sin(thetarM) * dt;

    //Anelastic walls
    xrabs = limitX.bound(xrabs);
    yrabs = limitY.bound(yrabs);
}

void Rocky::updateChickenPose(double v, double omega)
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

void Rocky::computeSensors(bool eat)
{
    vec2 chickenPosition = currentState.rows(span(x, y));

    currentState[energy] = maxEnergy.bound(currentState[energy] - 0.01);
    currentState[food] = 0;

    for (auto& spot : foodSpots)
    {
        if (norm(chickenPosition - spot) < 0.5)
        {
            currentState[food] = 1;

            if (eat)
            {
                currentState[energy] = maxEnergy.bound(currentState[energy] + 5);
            }

            break;
        }
    }
}

void Rocky::computeReward(Reward& reward)
{
    vec2 chickenPosition = currentState.rows(span(x, y));
    vec2 rockyRelPosition = currentState.rows(span(xr, yr));

    if (norm(rockyRelPosition) < 0.05)
    {
        reward[0] = -100;
        currentState.setAbsorbing(true);
    }
    else if (norm(chickenPosition) < 0.4 && currentState[energy] > 0)
    {
        reward[0] = currentState[energy];
        currentState.setAbsorbing(true);
    }
    else
    {
        reward[0] = 0;
        currentState.setAbsorbing(false);
    }
}

Rocky::Predictor::Predictor(double dt, Range limitX, Range limitY) :
    dt(dt), limitX(limitX), limitY(limitY)
{
    reset();
}

void Rocky::Predictor::reset()
{
    thetaM = 0;
    v = 0;
}

void Rocky::Predictor::saveLastValues(double thetaM, double v)
{
    this->thetaM = thetaM;
    this->v = v;
}

void Rocky::Predictor::predict(const DenseState& state, double& xhat, double& yhat, double& thetaDirhat)
{
    xhat = state[x] + v * cos(thetaM) * dt;
    yhat = state[y] + v * sin(thetaM) * dt;

    //Anelastic walls
    xhat = limitX.bound(xhat);
    yhat = limitY.bound(yhat);

    thetaDirhat = RangePi::wrap(atan2(yhat - (state[y] + state[yr]), xhat - (state[x] + state[xr])));
}

}
