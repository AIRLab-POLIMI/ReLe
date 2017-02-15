/*
 * rele,
 *
 *
 * Copyright (C) 2017 Davide Tateo
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

#include "rele/environments/WaterResources.h"
#include "rele/utils/ArmadilloPDFs.h"
#include "rele/utils/Range.h"
#include "rele/utils/RandomGenerator.h"

namespace ReLe
{

WaterResources::WaterResources()
{
	//TODO init parameters
}

void WaterResources::step(const DenseAction& action, DenseState& nextState,
                          Reward& reward)
{
	arma::vec& s = currentState;
    arma::vec v = s/delta;
    arma::vec V(2);
    V[up] = std::max(s[up] - maxCapacity[up], 0.0)/delta;
    V[dn] = std::max(s[dn] - maxCapacity[dn], 0.0)/delta;

    Range limitUp(v[up], V[up]);
    Range limitDown(v[dn], V[dn]);

    arma::vec r(2);
    r[up] = limitUp.bound(action[up]);
    r[dn] = limitUp.bound(action[dn]);

    arma::vec eps = mvnrand(mu, Sigma);

    currentState[up] += (eps[up] - r[up])*delta;
    currentState[dn] += (eps[dn] + r[up] - r[dn])*delta;


    nextState = currentState;
}

void WaterResources::getInitialState(DenseState& state)
{
	currentState[up] = RandomGenerator::sampleUniform(0, maxCapacity[up]);
	currentState[dn] = RandomGenerator::sampleUniform(0, maxCapacity[dn]);

	state = currentState;
}

void WaterResources::computeReward(Reward& reward, const arma::vec& r)
{
	arma::vec& s = currentState;
	arma::vec h = s / S;
	double P = computePowerGeneration(h[up], r[up]);

	reward[flo_up] = std::pow(std::max(h[up] - h_flo[up], 0.0), 2);
	reward[flo_dn] = std::pow(std::max(h[dn] - h_flo[dn], 0.0), 2);
	reward[irr] = std::pow(std::max(w_irr - r[dn], 0.0), 2);
	reward[hyd] = std::max(w_hyd - P, 0.0);
}

double WaterResources::computePowerGeneration(double h, double r)
{
	const double g = 9.81;
	double q = std::max(r - q_mef, 0.0);

	return eta*g*gamma_h20*h * q / (3.6e6);
}


}
