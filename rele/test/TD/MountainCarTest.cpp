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

#include "rele/environments/MountainCar.h"
#include "rele/core/Core.h"
#include "rele/algorithms/td/LinearSARSA.h"
#include "rele/algorithms/td/DenseSARSA.h"
#include "rele/approximators/basis/PolynomialFunction.h"
#include "rele/approximators/features/DenseFeatures.h"
#include "rele/approximators/basis/GaussianRbf.h"
#include "rele/approximators/basis/ConditionBasedFunction.h"
#include "rele/policy/q_policy/e_Greedy.h"
#include "rele/utils/FileManager.h"

#include "rele/approximators/regressors/q_regressors/QRegressorImplementation.h"

#include <string>

using namespace std;
using namespace ReLe;

int main(int argc, char *argv[])
{
    unsigned int nEpisodes = 100;
    MountainCar mdp(MountainCar::Ernst);

    BasisFunctions bVector = PolynomialFunction::generate(1, mdp.getSettings().statesNumber + 1);
    BasisFunctions basis = AndConditionBasisFunction::generate(bVector, 2, mdp.getSettings().actionsNumber);

    DenseFeatures phi(basis);

    e_GreedyApproximate policy;
    policy.setEpsilon(1);
    ConstantLearningRateDense alpha(0.1);
    LinearGradientSARSA agent(phi, policy, alpha);
    agent.setLambda(0.8);

    FileManager fm("mc", "fqi");
    fm.createDir();
    fm.cleanDir();

    unsigned int nExperiments = 20;
    for(unsigned int e = 0; e < nExperiments; e++)
    {
		auto&& core = buildCore(mdp, agent);
		std::string fileName = "mc_" + std::to_string(e) + ".log";
		core.getSettings().loggerStrategy = new WriteStrategy<FiniteAction, DenseState>(fm.addPath(fileName));

		for (int i = 0; i < nEpisodes; i++)
		{
			core.getSettings().episodeLength = 10000;
			cout << "Starting episode: " << i << endl;
			core.runEpisode();
		}
    }
}
