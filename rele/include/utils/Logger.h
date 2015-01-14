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

#ifndef INCLUDE_UTILS_LOGGER_H_
#define INCLUDE_UTILS_LOGGER_H_

#include "Basics.h"

#include <vector>
#include <iostream>

namespace ReLe
{

template<class ActionC, class StateC>
class Logger
{
public:
	void log(StateC& xn)
	{

	}

	void log(ActionC& u, StateC& x, Reward& r, unsigned int t)
	{

	}

	void printStatistics()
	{

	}

};

template<>
class Logger<FiniteAction, FiniteState>
{

public:
	void log(FiniteState& xn)
	{
		history.push_back(xn.getStateN());
	}

	void log(FiniteAction& u, FiniteState& xn, Reward& r, unsigned int t)
	{
		std::size_t x = history.back();
		history.push_back(xn.getStateN());

		std::cout << "t = " << t << ": (x = " << x << ", " << u << ") -> " << " ("
					<< xn << ", " << r << ")" << std::endl;

	}

	void printStatistics()
	{
		std::cout << std::endl << std::endl << "--- statistics ---" << std::endl
					<< std::endl;

		std::cout << "- State visits" << std::endl;
		//FIXME stati non noti a priori
		std::size_t totalVisits = history.size();
		std::size_t countedVisits = 0;
		for (std::size_t i = 0; countedVisits < totalVisits; i++)
		{
			std::size_t visits = std::count(history.begin(), history.end(), i);
			std::cout << "x(" << i << ") = " << visits << std::endl;
			countedVisits += visits;
		}

		std::cout << "- initial State" << std::endl << "x(t = 0): "
					<< history[0] << std::endl;
	}

private:
	std::vector<std::size_t> history;

};

}

#endif /* INCLUDE_UTILS_LOGGER_H_ */
