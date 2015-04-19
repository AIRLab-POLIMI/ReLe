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

#include "MountainCar.h"

#include "Core.h"
#include "td/LinearSARSA.h"
#include "basis/PolynomialFunction.h"

using namespace std;
using namespace ReLe;

int main(int argc, char *argv[])
{
    int episodes = 40;
    MountainCar mdp;

    BasisFunctions basis = PolynomialFunction::generate(1, mdp.getSettings().continuosStateDim + 1);
    DenseFeatures phi(basis);
    LinearApproximator approximator(3, phi);

    e_GreedyApproximate policy;
    LinearGradientSARSA agent(policy, approximator);

    Core<FiniteAction, DenseState> core(mdp, agent);

    for (int i = 0; i < episodes; i++)
    {
        core.getSettings().episodeLenght = 10000;
        cout << "starting episode" << endl;
        core.runEpisode();
    }

}