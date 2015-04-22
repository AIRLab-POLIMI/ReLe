/*
 * rele,
 *
 *
 * Copyright (C) 2015  Davide Tateo & Matteo Pirotta
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

#include "policy_search/PGPE/PGPE.h"
#include "policy_search/NES/NES.h"
#include "policy_search/REPS/REPS.h"
#include "DifferentiableNormals.h"
#include "Core.h"
#include "parametric/differentiable/LinearPolicy.h"
#include "PolicyEvalAgent.h"
#include "BasisFunctions.h"
#include "basis/IdentityBasis.h"
#include "basis/GaussianRbf.h"
#include "RandomGenerator.h"
#include "FileManager.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <random>
#include <cmath>
#include "Segway.h"

using namespace std;
using namespace ReLe;
using namespace arma;


/**
 *
 * argv[1] learning algorithm name (pgpe, nes, enes, reps) -> pgpe and nes requires the distribution type
 * argv[2] distribution type (gauss, chol, diag, log)
 * argv[2/3] # updates
 * argv[3/4] # policies per update
 * argv[4/5] learning rate for updates
 * argv[5/6] stepType ("constant", "adaptive")
 *
 */
int main(int argc, char *argv[])
{

    FileManager fm("segway", "BBO");
    fm.createDir();
    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);


    Segway mdp;

    BasisFunctions basis = GaussianRbf::generate({3,3,3}, {-M_PI/18, M_PI/18, -10, 10, -7,7});
    DenseFeatures phi(basis);
    DetLinearPolicy<DenseState> policy(phi);
    //---

    PolicyEvalAgent<DenseAction, DenseState> agent(policy);
    Core<DenseAction, DenseState> core(mdp, agent);
    WriteStrategy<DenseAction, DenseState> collection(fm.addPath("seqway_final.log"),
            WriteStrategy<DenseAction, DenseState>::TRANS,
            true /*delete file*/);
    core.getSettings().loggerStrategy = &collection;
    core.getSettings().episodeLenght = mdp.getSettings().horizon;
    core.getSettings().testEpisodeN = 1;
    core.runTestEpisodes();

    return 0;
}
