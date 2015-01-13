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
#include <iostream>
#include <string>

namespace ReLe
{
class Action
{
public:
	inline virtual std::string to_str() const
	{
		return "action";
	}

	inline virtual ~Action()
	{

	}
};

class FiniteAction: public Action
{
public:
	FiniteAction()
	{
		actionN = 0;
	}

	inline int getActionN() const
	{
		return actionN;
	}

	inline void setActionN(int actionN)
	{
		this->actionN = actionN;
	}

	inline virtual std::string to_str() const
	{
		return "u = " + std::to_string(actionN);
	}

	inline virtual ~FiniteAction()
	{

	}

private:
	int actionN;
};

class State
{
public:
	State() :
				absorbing(false)
	{

	}

	inline bool isAbsorbing() const
	{
		return absorbing;
	}

	inline virtual std::string to_str() const
	{
		return "state";
	}

	inline virtual ~State()
	{

	}

private:
	bool absorbing;
};

class FiniteState: public State
{
public:
	FiniteState()
	{
		stateN = 0;
	}

	inline std::size_t getStateN() const
	{
		return stateN;
	}

	inline void setStateN(std::size_t stateN)
	{
		this->stateN = stateN;
	}

	inline virtual std::string to_str() const
	{
		return "x = " + std::to_string(stateN);
	}

	inline virtual ~FiniteState()
	{

	}

private:
	std::size_t stateN;
};

typedef std::vector<double> Reward;

inline std::ostream& operator<<(std::ostream& os, const Action& action)
{
	os << action.to_str();
	return os;
}

inline std::ostream& operator<<(std::ostream& os, const State& state)
{
	os << state.to_str();
	return os;
}

inline std::ostream& operator<<(std::ostream& os, const Reward& reward)
{
	os << "[ ";
	for(auto r : reward)
		os << r << " ";
	os << "]";
	return os;
}


}

#endif /* BASICS_H_ */
