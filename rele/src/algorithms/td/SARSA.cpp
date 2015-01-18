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
	alpha = 0.2;
	eps = 0.15;
}

void SARSA::initEpisode(const FiniteState& state,  FiniteAction& action)
{
	sampleAction(state, action);
}

void SARSA::sampleAction(const FiniteState& state, FiniteAction& action)
{
	x = state.getStateN();
	u = policy(x);

	action.setActionN(u);
}

void SARSA::step(const Reward& reward, const FiniteState& nextState, FiniteAction& action)
{
	size_t xn = nextState.getStateN();
	unsigned int un = policy(xn);
	double r = reward[0];

	double delta = r + task.gamma * Q(xn, un) - Q(x, u);
	Q(x, u) = Q(x, u) + alpha * delta;

	//update action and state
	x = xn;
	u = un;

	//set next action
	action.setActionN(u);
}

void SARSA::endEpisode()
{
	//print statistics
	printStatistics();
}

void SARSA::endEpisode(const Reward& reward)
{
	//Last update
	double r = reward[0];
	double delta = r - Q(x, u);
	Q(x, u) = Q(x, u) + alpha * delta;

	//print statistics
	printStatistics();
}

SARSA::~SARSA()
{

}

unsigned int SARSA::policy(size_t x)
{
	unsigned int un;

	const rowvec& Qx = Q.row(x);

	if (RandomGenerator::sampleEvent(eps))
		un = RandomGenerator::sampleUniformInt(0, Q.n_cols - 1);
	else
		Qx.max(un);

	return un;
}

void SARSA::printStatistics()
{
	//TODO dentro la classe o altrove???
	cout << endl << endl << "### SARSA ###";

	cout << endl << endl << "--- Parameters --"
				<< endl << endl;
	cout << "gamma: " << gamma << endl;
	cout << "alpha: " << alpha << endl;
	cout << "eps: " << eps << endl;

	cout << endl << endl << "--- Learning results ---"
				<< endl << endl;

	cout << "- Action-value function" << endl;
	for (unsigned int i = 0; i < Q.n_rows; i++)
		for (unsigned int j = 0; j < Q.n_cols; j++)
		{
			cout << "Q(" << i << ", " << j << ") = " << Q(i, j) << endl;
		}
	cout << "- Policy" << endl;
	for (unsigned int i = 0; i < Q.n_rows; i++)
	{
		unsigned int policy;
		Q.row(i).max(policy);
		cout << "policy(" << i << ") = " << policy << endl;
	}

}

SARSA_lambda::SARSA_lambda(double lambda) :
			lambda(lambda)
{

}

void SARSA_lambda::initEpisode(const FiniteState& state, FiniteAction& action)
{

}

void SARSA_lambda::sampleAction(const FiniteState& state, FiniteAction& action)
{

}

void SARSA_lambda::step(const Reward& reward, const FiniteState& nextState, FiniteAction& action)
{

}

void SARSA_lambda::endEpisode()
{

}

void SARSA_lambda::endEpisode(const Reward& reward)
{

}

SARSA_lambda::~SARSA_lambda()
{

}

}
