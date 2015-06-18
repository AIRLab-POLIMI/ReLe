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

#include "RockyOptions.h"
#include "Rocky.h"

#include "ModularRange.h"

#include <cassert>

using namespace arma;
using namespace std;

namespace ReLe
{

using sc=Rocky::StateComponents;

RockyOption::RockyOption() : maxV(1), dt(0.01)
{

}

arma::vec RockyOption::wayPointPolicy(const arma::vec& state, const arma::vec& target)
{
    double ox = target[0];
    double oy = target[1];
    double waypointDir = atan2(oy - state[sc::y], ox - state[sc::x]);
    double deltaTheta = angularDistance(state, target);
    double omega = deltaTheta / dt;
    double v;

    vec deltaPos(2);
    deltaPos[0] = ox - state[sc::x];
    deltaPos[1] = oy - state[sc::y];

    if (abs(deltaTheta) > M_PI / 2 || norm(deltaPos) < 0.01)
    {
        v = 0;
    }
    else if (abs(deltaTheta) > M_PI / 4)
    {
        v = maxV / 2;
    }
    else
    {
        v = maxV;
    }

    vec pi(3);
    pi[0] = v;
    pi[1] = omega;
    pi[2] = 0;

    return pi;
}

double RockyOption::angularDistance(const arma::vec& state, const arma::vec& target)
{
    double waypointDir = atan2(target[1] - state[sc::y], target[0] - state[sc::x]);
    return RangePi::wrap(waypointDir - state[sc::theta]);
}


double RockyOption::rockyRelRotation(const arma::vec& state)
{
    return RangePi::wrap(atan2(state[sc::yr],state[sc::xr]));
}

bool Eat::canStart(const arma::vec& state)
{
    return state[sc::energy] < 100 && state[sc::food] == 1.0;
}

double Eat::terminationProbability(const DenseState& state)
{
    if(state[sc::energy] >= 100)
        return 1;

    double distP = min(1.0, 0.3/norm(state(span(sc::xr, sc::yr))));
    double energyP = state[sc::energy] / 100;

    return std::max(distP, energyP);
}

void Eat::operator ()(const DenseState& state, DenseAction& action)
{
    assert(state[sc::food] == 1.0);
    vec pi(3);

    pi[0] = 0;
    pi[1] = 0;
    pi[2] = 1;

    vec& x = action;
    x = pi;
}

bool Home::canStart(const arma::vec& state)
{
    return state[sc::energy] > 0;
}

Home::Home() : home(2)
{
    home[0] = 0;
    home[1] = 0;
}

double Home::terminationProbability(const DenseState& state)
{
    if(state[sc::energy] == 0 || norm(state(span(sc::x, sc::y)) - home) < 0.4)
        return 1;
    else
        return 0;

}

void Home::operator ()(const DenseState& state, DenseAction& action)
{
    arma::vec pi = wayPointPolicy(state, home);

    vec& x = action;
    x = pi;
}

Feed::Feed() : spot(2)
{
    spot[0] = 5;
    spot[1] = 0;
}


bool Feed::canStart(const arma::vec& state)
{
    return state[sc::energy] < 100 && norm(state(span(sc::x,sc::y))- spot) > 0.5;
}

double Feed::terminationProbability(const DenseState& state)
{
    if(norm(state(span(sc::x, sc::y)) - spot) < 0.5)
        return 1;
    else
        return min(1.0, 0.4/norm(state(span(sc::xr,sc::yr))));
}

void Feed::operator ()(const DenseState& state, DenseAction& action)
{
    arma::vec pi = wayPointPolicy(state, spot);

    vec& x = action;
    x = pi;
}

bool Escape1::canStart(const arma::vec& state)
{
    return true;
}

double Escape1::terminationProbability(const DenseState& state)
{
    double distanceP = min(1.0, norm(state(span(sc::xr, sc::yr))));
    double angularDistance = abs(RangePi::wrap(state[sc::theta]-state[sc::thetar]));
    double angleP = angularDistance/M_1_PI;
    return max(distanceP, angleP);
}

void Escape1::operator ()(const DenseState& state, DenseAction& action)
{
    action.resize(3);
    action[0] = maxV;
    action[1] = M_PI;
    action[2] = 0;
}

bool Escape2::canStart(const arma::vec& state)
{
    return true;
}

double Escape2::terminationProbability(const DenseState& state)
{
    double distanceP = min(1.0, norm(state(span(sc::xr, sc::yr))));
    double angularDistance = abs(RangePi::wrap(state[sc::theta]-state[sc::thetar]));
    double angleP = angularDistance/M_1_PI;
    return max(distanceP, angleP);
}

void Escape2::operator ()(const DenseState& state, DenseAction& action)
{
    action.resize(3);
    action[0] = maxV;
    action[1] = -M_PI;
    action[2] = 0;
}

bool Escape3::canStart(const arma::vec& state)
{
    return true;
}

double Escape3::terminationProbability(const DenseState& state)
{
    double distanceP = min(1.0, norm(state(span(sc::xr, sc::yr))));
    double angularDistance = abs(RangePi::wrap(state[sc::theta]-state[sc::thetar]));
    double angleP = angularDistance/M_1_PI;
    return max(distanceP, angleP);
}

void Escape3::operator ()(const DenseState& state, DenseAction& action)
{
    action.resize(3);
    action[0] = maxV;
    action[1] = state[sc::thetar]/dt;
    action[2] = 0;
}





}
