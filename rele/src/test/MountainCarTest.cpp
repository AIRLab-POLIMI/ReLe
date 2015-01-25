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

#include "Core.h"
#include "MountainCar.h"
#include "td/LinearSARSA.h"
#include "basis/GaussianRBF.h"

using namespace std;

int main(int argc, char *argv[])
{
	int episodes = 1;
	ReLe::MountainCar mdp;

	ReLe::BasisFunctions bf;
	ReLe::LinearApproximator approximator(3, bf);
	ReLe::LinearGradientSARSA agent(approximator);

	ReLe::Core<ReLe::FiniteAction, ReLe::DenseState> core(mdp, agent);

	for (int i = 0; i < episodes; i++)
	{
		core.getSettings().episodeLenght = 10000;
		core.getSettings().logTransitions = false;
		cout << "starting episode" << endl;
		core.runEpisode();
	}
}
