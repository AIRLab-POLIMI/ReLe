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

#include "Utils.h"

#include <cassert>

using namespace arma;
using namespace std;

namespace ReLe
{

RockyOption::RockyOption() : maxV(1), dt(0.01)
{

}

arma::vec RockyOption::wayPointPolicy(const arma::vec& state, const arma::vec& target)
{
    double ox = target[0];
    double oy = target[1];
    double waypointDir = atan2(oy - state[y], ox - state[x]);
    double deltaTheta = angularDistance(state, target);
    double omega = deltaTheta / dt;
    double v;

    vec deltaPos(2);
    deltaPos[0] = ox - state[x];
    deltaPos[1] = oy - state[y];

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
    double waypointDir = atan2(target[1] - state[y], target[0] - state[x]);
    return utils::wrapToPi(waypointDir - state[theta]);
}


double RockyOption::rockyRelRotation(const arma::vec& state)
{
    return utils::wrapToPi(atan2(state[yr],state[xr]));
}

bool Eat::canStart(const arma::vec& state)
{
    return state[energy] < 100 && state[food] == 1.0;
}

double Eat::terminationProbability(const DenseState& state)
{
    if(state[energy] >= 100)
        return 1;

    double distP = min(1.0, norm(state(span(xr, yr))));
    double energyP = state[energy] / 100;

    return std::max(distP, energyP);
}

void Eat::operator ()(const DenseState& state, DenseAction& action)
{
    assert(state[food] == 1);
    vec pi(3);

    pi[0] = 0;
    pi[1] = 0;
    pi[2] = 1;

    vec& x = action;
    x = pi;
}

bool Home::canStart(const arma::vec& state)
{
    return state[energy] > 0;
}

Home::Home() : home(2)
{
    home[0] = 0;
    home[1] = 0;
}

double Home::terminationProbability(const DenseState& state)
{
    if(state[energy] == 0 || norm(state(span(x, y)) - home) < 0.4)
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
    return state[energy] < 100 && norm(state(span(x,y))- spot) > 0.5;
}

double Feed::terminationProbability(const DenseState& state)
{
    if(norm(state(span(x, y)) - spot) < 0.5)
        return 1;
    else
        return min(1.0, 0.4/norm(state(span(xr,yr))));
}

void Feed::operator ()(const DenseState& state, DenseAction& action)
{
    arma::vec pi = wayPointPolicy(state, spot);

    vec& x = action;
    x = pi;
}

bool Escape1::canStart(const arma::vec& state)
{
    return norm(state(span(xr, yr))) < 1;
}

double Escape1::terminationProbability(const DenseState& state)
{
    return min(1.0, 0.8*norm(state(span(xr, yr))));
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
    return norm(state(span(xr, yr))) < 1;
}

double Escape2::terminationProbability(const DenseState& state)
{
    return min(1.0, 0.8*norm(state(span(xr, yr))));
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
    return norm(state(span(xr, yr))) < 1;
}

double Escape3::terminationProbability(const DenseState& state)
{
    return min(1.0, 0.8*norm(state(span(xr, yr))));
}

void Escape3::operator ()(const DenseState& state, DenseAction& action)
{
    action.resize(3);
    action[0] = maxV;
    action[1] = state[thetar]/dt;
    action[2] = 0;
}





}
