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

#ifndef FINITEMDP_H_
#define FINITEMDP_H_

#include "Basics.h"
#include "Envirorment.h"

#include <vector>

namespace ReLe
{

class FiniteMDP: public Envirorment<FiniteAction, FiniteState>
{
//TODO: levare tutti sti template del cavolo per poter compilare un esempuio statico.
//TODO: magari usare una libreria di matrici... o classi apposta.
public:
	template<int statesNumber, int actionsNumber>
	FiniteMDP(const double (&Pdata)[actionsNumber][statesNumber][statesNumber],
				const double (&Rdata)[statesNumber][2], bool isFiniteHorizon, double gamma = 1.0) :
				Envirorment()
	{
		initP(Pdata);
		initR(Rdata);

		EnvirormentSettings& task = getWritableSettings();
		task.isAverageReward = false;
		task.isDiscreteActions = true;
		task.isDiscreteStates = true;
		task.isEpisodic = false;
		task.isFiniteHorizon = isFiniteHorizon;
		task.gamma = gamma;
	}

	virtual void step(const FiniteAction& action, FiniteState& nextState,
				Reward& reward);
	virtual void getInitialState(FiniteState& state);

private:
	std::vector<std::vector<std::vector<double>>>P;
	std::vector<std::vector<double>> R;
	FiniteState currentState;

private:
	template<int statesNumber, int actionsNumber>
	void initP(const double (&Pdata)[actionsNumber][statesNumber][statesNumber])
	{
		P.resize(actionsNumber);
		for (std::size_t k = 0; k < actionsNumber; k++)
		{
			P[k].resize(statesNumber);
			for (std::size_t i = 0; i < statesNumber; i++)
			{
				P[k][i].resize(statesNumber);
				for (std::size_t j = 0; j < statesNumber; j++)
				{
					P[k][i][j] = Pdata[k][i][j];
				}
			}
		}
	}

	template<int statesNumber>
	void initR(const double (&Rdata)[statesNumber][2])
	{

		R.resize(statesNumber);
		for (std::size_t i = 0; i < statesNumber; i++)
		{
			R[i].resize(2);
			R[i][0] = Rdata[i][0]; //mean
			R[i][1] = Rdata[i][1];//variance
		}

	}
};

}

#endif /* FINITEMDP_H_ */
