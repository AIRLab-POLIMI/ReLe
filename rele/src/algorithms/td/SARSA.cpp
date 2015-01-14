/*
 * rele,
 *
 *
 * Copyright (C) 2015 Davide Tateo
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

#include "td/SARSA.h"
#include "RandomGenerator.h"

using namespace std;
using namespace arma;

namespace ReLe
{

SARSA::SARSA(size_t statesN, size_t actionN) :
			Q(statesN, actionN, fill::zeros)
{
	x = 0;
	u = 0;

	//Default algorithm parameters
	alpha = 1;
	gamma = 0.9;

	//time step
	t = 1;
	actionReady = false;
}

void SARSA::initEpisode()
{
	t = 1;
}

void SARSA::sampleAction(const FiniteState& state, FiniteAction& action)
{
	x = state.getStateN();

	if (!actionReady)
	{
		u = policy(x);
	}

	action.setActionN(u);
	actionReady = false;
}

void SARSA::step(const Reward& reward, const FiniteState& nextState)
{
	size_t xn = nextState.getStateN();
	unsigned int un = policy(xn);
	double r = reward[0];

	double delta = r + gamma * Q(xn, un) - Q(x, u);
	Q(x, u) = Q(x, u) + alpha * delta;

	cout << "Q(" << x << ", " << u << ") = " << Q(x, u) << endl;

	//update action
	u = un;
	actionReady = true;

	t++;
}

void SARSA::endEpisode(const Reward& reward)
{
	//TODO che ci metto qui?
	// Per ora stampo la action-value function...
	for (unsigned int i = 0; i < Q.n_rows; i++)
		for (unsigned int j = 0; j < Q.n_cols; j++)
		{
			cout << "Q(" << i << ", " << j << ") = " << Q(i, j) << endl;
		}

	//E la policy...
	for (unsigned int i = 0; i < Q.n_rows; i++)
	{
		unsigned int policy;
		Q.row(i).max(policy);
		cout << "policy(" << i << ") = " << policy << endl;
	}

}

SARSA::~SARSA()
{

}

unsigned int SARSA::policy(size_t x)
{
	unsigned int un;

	const rowvec& Qx = Q.row(x);

	double eps = 0.15;

	if (RandomGenerator::sampleEvent(eps))
		un = RandomGenerator::sampleUniformInt(0, Q.n_cols - 1);
	else
		Qx.max(un);

	return un;
}

SARSA_lambda::SARSA_lambda(double lambda) :
			lambda(lambda)
{

}

void SARSA_lambda::initEpisode()
{

}

void SARSA_lambda::sampleAction(const FiniteState& state, FiniteAction& action)
{

}

void SARSA_lambda::step(const Reward& reward, const FiniteState& nextState)
{

}

void SARSA_lambda::endEpisode(const Reward& reward)
{

}

SARSA_lambda::~SARSA_lambda()
{

}

}
