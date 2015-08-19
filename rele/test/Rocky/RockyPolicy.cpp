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

#include "RockyPolicy.h"

#include "ModularRange.h"

using namespace arma;
using namespace std;

namespace ReLe
{

RockyPolicy::RockyPolicy(double dt) : maxV(1), dt(dt)
{
    w.zeros(PARAM_SIZE);
}

arma::vec RockyPolicy::operator() (const arma::vec& state)
{
    //compute objective
    auto objective = computeObjective(state);

    switch(objective)
    {
    case eat:
        return eatPolicy();

    case home:
        return homePolicy(state);

    case feed:
        return feedPolicy(state);

    case escape:
        return escapePolicy(state);

    default:
        std::cout << "Error!" << std::endl;
        return vec(3, fill::zeros);

    }

}


double RockyPolicy::operator() (const arma::vec& state, const arma::vec& action)
{
    return 0;
}

RockyPolicy::Objective RockyPolicy::computeObjective(const arma::vec& state)
{
    const vec& rockyDistance = state(span(xr, yr));
    if(norm(rockyDistance) < abs(w[escapeThreshold]))
    {
        //std::cout << "escape" << std::endl;
        return escape;
    }
    else if(state[food] == 1 && state[energy] < min(2*abs(w[energyThreshold]), 100.0))
    {
        //std::cout << "eat" << std::endl;
        return eat;
    }
    else if(state[energy] < abs(w[energyThreshold]))
    {
        //std::cout << "home" << std::endl;
        return feed;
    }
    else
    {
        //std::cout << "feed" << std::endl;
        return home;
    }
}

vec RockyPolicy::eatPolicy()
{
    vec pi(3);

    pi[0] = 0;
    pi[1] = 0;
    pi[2] = 1;

    return pi;
}

arma::vec RockyPolicy::homePolicy(const arma::vec& state)
{
    return wayPointPolicy(state, 0, 0);
}

arma::vec RockyPolicy::feedPolicy(const arma::vec& state)
{
    return wayPointPolicy(state, 5, 0);
}

arma::vec RockyPolicy::escapePolicy(const arma::vec& state)
{
    vec we = w(span(escapeParamsStart, escapeParamsEnd));
    mat phi(12, 3, fill::zeros);

    phi(span(0, 2), span(0)) = state(span(x, theta));
    phi(span(3, 5), span(0)) = state(span(xr, thetar));

    phi(span(5, 7), span(1)) = state(span(x, theta));
    phi(span(8, 10), span(1)) = state(span(xr, thetar));

    vec pi = phi.t()*we;

    return pi;
}

arma::vec RockyPolicy::wayPointPolicy(const arma::vec& state, double ox, double oy)
{
    double waypointDir = atan2(oy - state[y], ox - state[x]);
    double deltaTheta = RangePi::wrap(waypointDir - state[theta]);
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

}
