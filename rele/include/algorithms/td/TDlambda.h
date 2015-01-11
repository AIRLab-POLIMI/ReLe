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

#ifndef TDLAMBDA_H_
#define TDLAMBDA_H_

#include "Agent.h"

namespace ReLe
{

class TD_lambda : public Agent<FiniteAction, FiniteState>
{
public:
	TD_lambda(double lambda);
	virtual void initEpisode();
	virtual void sampleAction(const FiniteState& state, FiniteAction& action);
	virtual void step(const Reward& reward, const FiniteState& nextState);
	virtual void endEpisode(const Reward& reward);

	virtual ~TD_lambda();

private:
	double lambda;
};

class TD_0 : public TD_lambda
{
public:
	virtual void initEpisode();
	virtual void sampleAction(const FiniteState& state, FiniteAction& action);
	virtual void step(const Reward& reward, const FiniteState& nextState);
	virtual void endEpisode(const Reward& reward);

	virtual ~TD_0();
};

}
#endif /* TDLAMBDA_H_ */
