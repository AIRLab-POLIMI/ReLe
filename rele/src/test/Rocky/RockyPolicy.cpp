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

using namespace arma;

namespace ReLe
{


arma::vec RockyPolicy::operator() (const arma::vec& state)
{
    //robot pose
    double x = state[0];
    double y = state[1];
    double theta = state[2];

    //robot sensors
    double energy = state[3];
    double food = state[4];

    //rocky pose
    double xr = state[5];
    double yr = state[6];
    double thetar = state[7];

    //compute objective
    auto objective = computeObjective(state);

    switch(objective)
    {
    	case eat:
    		return eatPolicy();


    }

}


double RockyPolicy::operator() (const arma::vec& state, const arma::vec& action)
{
	return 0;
}

RockyPolicy::Objective RockyPolicy::computeObjective(const arma::vec& state)
{
	const vec& rockyDistance = state(span(xr, yr));
	if(norm(rockyDistance) < w[escapeThreshold])
	{
		return escape;
	}
	else if(state[energy] > w[homeThreshold])
	{
		return home;
	}
	else if(state[food] == 1)
	{
		return eat;
	}
	else
	{
		return feed;
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

}
