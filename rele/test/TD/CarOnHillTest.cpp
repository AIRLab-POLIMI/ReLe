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

#include "rele/core/Core.h"

#include "rele/algorithms/td/LinearSARSA.h"
#include "rele/algorithms/td/DenseSARSA.h"

#include "rele/environments/CarOnHill.h"

#include "rele/approximators/features/DenseFeatures.h"
#include "rele/approximators/features/TilesCoder.h"

#include "rele/approximators/tiles/BasicTiles.h"
#include "rele/approximators/basis/PolynomialFunction.h"
#include "rele/approximators/basis/ConditionBasedFunction.h"

#include "rele/policy/q_policy/e_Greedy.h"
#include "rele/utils/FileManager.h"
#include "rele/utils/Range.h"


#include <string>

using namespace std;
using namespace ReLe;

int main(int argc, char *argv[])
{
	FileManager fm("mc", "LinearSARSA");
	fm.createDir();
	fm.cleanDir();

    unsigned int nEpisodes = 10000;
    CarOnHill mdp;

    unsigned int tilesN = 9;
    unsigned int actionsN = mdp.getSettings().actionsNumber;
    Range xRange(-1, 1);
    Range vRange(-3, 3);

    auto* tiles = new BasicTiles({xRange, vRange, Range(-0.5, 1.5)},{tilesN, tilesN, actionsN});

    DenseTilesCoder phi(tiles);

    e_GreedyApproximate policy;
    policy.setEpsilon(0.1);
    ConstantLearningRateDense alpha(0.1);
    LinearGradientSARSA agent(phi, policy, alpha);
    agent.setLambda(0.9);


    auto&& core = buildCore(mdp, agent);
    core.getSettings().episodeLength = mdp.getSettings().horizon;
    core.getSettings().episodeN = nEpisodes;
    core.runEpisodes();


    policy.setEpsilon(0.0);
    core.getSettings().testEpisodeN = 1;
    core.getSettings().loggerStrategy = new PrintStrategy<FiniteAction, DenseState>();
    core.runTestEpisodes();

    cout << "Objective Function =" << core.runEvaluation().t() << endl;

}
