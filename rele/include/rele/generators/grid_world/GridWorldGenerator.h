/*
 * rele,
 *
 *
 * Copyright (C) 2015 Davide Tateo & Matteo Pirotta
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

#ifndef GRIDWORLDGENERATOR_H_
#define GRIDWORLDGENERATOR_H_

#include "FiniteMDP.h"

namespace ReLe
{

class GridWorldGenerator
{
public:
	GridWorldGenerator();
	void load(const std::string& path);
	FiniteMDP getMPD(double gamma);

private:
	void assignStateNumbers(std::vector<std::vector<char>>& matrix,
				std::vector<std::vector<int>>& stateNMatrix, std::size_t i,
				std::size_t j);
	void handleChar(std::vector<std::vector<char>>& matrix, std::size_t i,
				std::size_t j);

private:
	arma::cube P;
	arma::cube R;

	size_t stateN;
	size_t currentState;
	int actionN;

};

}
#endif /* GRIDWORLDGENERATOR_H_ */
