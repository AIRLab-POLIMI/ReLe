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

#ifndef E_GREEDY_H_
#define E_GREEDY_H_

#include "Policy.h"

namespace ReLe
{

class e_Greedy: public ActionValuePolicy<FiniteState>
{
public:
	 virtual int operator() (int state);
	 virtual double operator() (int state, int action);

	 virtual ~e_Greedy();
};

class e_GreedyApproximate: public ActionValuePolicy<DenseState>
{
public:
	 virtual int operator() (arma::vec& state);
	 virtual double operator() (arma::vec& state, int action);

	 virtual ~e_GreedyApproximate();
};

}
#endif /* E_GREEDY_H_ */
