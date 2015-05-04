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

arma::vec RockyOption::wayPointPolicy(const arma::vec& state, double ox, double oy)
{
    double waypointDir = atan2(oy - state[y], ox - state[x]);
    double deltaTheta = utils::wrapToPi(waypointDir - state[theta]);
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

bool Eat::canStart(const DenseState& state)
{
    return state[energy] < 100 && state[food] == 1;
}

double Eat::terminationProbability(const DenseState& state)
{

    if(state[food] == 0)
        return 1;
    if(state[energy] >= 100)
        return 1;
    else if(norm(state(span(xr, yr))) < 0.2)
        return 1;
    else
        return state[energy] / 100.0;
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

bool Home::canStart(const DenseState& state)
{
    return state[energy] > 0;
}

double Home::terminationProbability(const DenseState& state)
{
    if(norm(state(span(xr, yr))) < 0.2)
        return 1;
    else
        return 0;

}

void Home::operator ()(const DenseState& state, DenseAction& action)
{
    arma::vec pi = wayPointPolicy(state, 0, 0);

    vec& x = action;
    x = pi;
}

Feed::Feed() : spot(2)
{
    spot[0] = 5;
    spot[1] = 0;
}


bool Feed::canStart(const DenseState& state)
{
    return state[energy] < 100;
}

double Feed::terminationProbability(const DenseState& state)
{
    if(norm(state(span(x, y)) - spot) < 0.5)
        return 1;
    else if(norm(state(span(xr, yr))) < 0.2)
        return 1;
    else
        return 0;
}

void Feed::operator ()(const DenseState& state, DenseAction& action)
{
    arma::vec pi = wayPointPolicy(state, spot[0], spot[1]);

    vec& x = action;
    x = pi;

    //cout << action << endl;
}

bool Escape::canStart(const DenseState& state)
{
    return true;
}

double Escape::terminationProbability(const DenseState& state)
{
    if(norm(state(span(xr, yr))) > 1)
        return 1;
    else
        return norm(state(span(xr, yr)));
}

void Escape::operator ()(const DenseState& state, DenseAction& action)
{
    vec we =
    {
        11.3530,
        -24.0812,
        -5.2888,
        6.3575,
        11.5793,
        -6.1634,
        5.3837,
        -5.9982,
        -5.0773,
        -9.1568,
        5.1742,
        -3.8392
    };

    mat phi(12, 3, fill::zeros);

    phi(span(0, 2), span(0)) = state(span(x, theta));
    phi(span(3, 5), span(0)) = state(span(xr, thetar));

    phi(span(5, 7), span(1)) = state(span(x, theta));
    phi(span(8, 10), span(1)) = state(span(xr, thetar));

    vec pi = phi.t()*we;

    vec& x = action;
    x = pi;
}







}
