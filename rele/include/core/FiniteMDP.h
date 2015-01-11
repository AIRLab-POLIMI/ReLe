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

class FiniteMDP : public Envirorment<FiniteAction, FiniteState>
{
//TODO: levare tutti sti template del cavolo per poter compilare un esempuio statico.
//TODO: magari usare una libreria di matrici... o classi apposta.
public:
	template<int statesNumber, int actionsNumber>
	FiniteMDP(const double (&Pdata)[actionsNumber][statesNumber][statesNumber],
				const double (&Rdata)[actionsNumber][statesNumber][2]) : Envirorment()
	{
		initP(Pdata);
		initR(Rdata);
	}

	virtual bool step(const FiniteAction& action, FiniteState& nextState,
					Reward& reward);
	virtual void getInitialState(FiniteState& state);


private:
	std::vector<std::vector<std::vector<double>>> P;
	std::vector<std::vector<std::vector<double>>> R;
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

	template<int statesNumber, int actionsNumber>
	void initR(const double (&Rdata)[actionsNumber][statesNumber][2])
	{
		R.resize(actionsNumber);
		for (std::size_t k = 0; k < actionsNumber; k++)
		{
			R[k].resize(statesNumber);
			for (std::size_t i = 0; i < statesNumber; i++)
			{
				R[k][i].resize(2);
				R[k][i][0] = Rdata[k][i][0]; //mean
				R[k][i][1] = Rdata[k][i][1]; //variance
			}
		}
	}
};

}


#endif /* FINITEMDP_H_ */
