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

#ifndef BASICS_H_
#define BASICS_H_

#include <vector>

namespace ReLe
{
class Action
{

};

class FiniteAction : public Action
{
public:
	inline int getActionN() const
	{
		return actionN;
	}

private:
	int actionN;
};

class State
{
public:
	inline bool isAbsorbing() const
	{
		return absorbing;
	}

private:
	bool absorbing;
};

class FiniteState : public State
{
public:
	inline std::size_t getStateN() const
	{
		return stateN;
	}

	inline void setStateN(std::size_t stateN)
	{
		this->stateN = stateN;
	}

private:
	std::size_t stateN;
};

typedef std::vector<double> Reward;

}

#endif /* BASICS_H_ */
