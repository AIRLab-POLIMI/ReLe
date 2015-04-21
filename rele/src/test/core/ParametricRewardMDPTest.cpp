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
#include "BasisFunctions.h"
#include "basis/IdentityBasis.h"
#include "LinearApproximator.h"
#include "ParametricRewardMDP.h"

#include "FileManager.h"
#include "ConsoleManager.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <random>
#include <cmath>
#include "DeepSeaTreasure.h"
#include "LQR.h"

using namespace std;
using namespace ReLe;
using namespace arma;

int main(int argc, char *argv[])
{
    FileManager fm("IRL", "basicTest");
    fm.createDir();
    fm.cleanDir();

    LQR mdp1(1, 1);
    DeepSeaTreasure mdp2;


    IdentityBasis* pf1 = new IdentityBasis(0);
    DenseFeatures basis1(pf1);
    LinearApproximator regressor1(basis1);

    IdentityBasis* pf2 = new IdentityBasis(0);
    DenseFeatures basis2(pf2);
    LinearApproximator regressor2(basis1);

    ParametricRewardMDP<DenseAction, DenseState> prMDP1(mdp1, regressor1);
    ParametricRewardMDP<FiniteAction, DenseState> prMDP2(mdp2, regressor2);


    /*ReLe::Core<DenseAction, DenseState> core(mdp, agent);

    core.getSettings().loggerStrategy = new WriteStrategy<DenseAction,
    DenseState>(fm.addPath("agent.log"),
                WriteStrategy<DenseAction, DenseState>::AGENT);

    ConsoleManager console(episodes, 1);
    for (int i = 0; i < episodes; i++)
    {
        core.getSettings().episodeLenght = mdp.getSettings().horizon;
        console.printProgress(i);
        core.runEpisode();
    }

    delete core.getSettings().loggerStrategy;*/

    return 0;
}
