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

#include "rele/core/BatchCore.h"
#include "rele/policy/parametric/differentiable/NormalPolicy.h"
#include "rele/algorithms/batch/policy_search/gradient/OffPolicyGradientAlgorithm.h"
#include "rele/approximators/basis/IdentityBasis.h"
#include "rele/approximators/features/DenseFeatures.h"
#include "rele/utils/RandomGenerator.h"
#include "rele/utils/FileManager.h"
#include "rele/environments/NLS.h"

using namespace std;
using namespace ReLe;
using namespace arma;

int main(int argc, char *argv[])
{
    FileManager fm("Offpolicy", "nls");
    fm.createDir();
    fm.cleanDir();

    NLS mdp;
    //with these settings
    //max in ( many optimal points ) -> J = 8.5
    //note that there are multiple optimal solutions
    //e.g.
    //-3.2000    8.8000    8.4893
    //-3.2000    9.3000    8.4959
    //-3.2000    9.5000    8.4961
    //-3.4000   10.0000    8.5007
    //-3.2000    9.4000    8.5020
    //-3.1000    8.8000    8.5028
    //-3.4000    9.7000    8.5041
    //-3.0000    8.1000    8.5205
    //-2.9000    7.7000    8.5230
    //-3.1000    9.1000    8.5243
    //-2.8000    7.3000    8.5247
    int dim = mdp.getSettings().stateDimensionality;

    // define policy
    BasisFunctions basis = IdentityBasis::generate(dim);
    DenseFeatures phi(basis);
    arma::vec wB(2);
    wB(0) = -0.57;
    wB(1) = 0.5;

    BasisFunctions stdBasis = IdentityBasis::generate(dim);
    DenseFeatures stdPhi(stdBasis);
    arma::vec stdWeights(stdPhi.rows());
    stdWeights.fill(0.5);

    NormalStateDependantStddevPolicy behavioral(phi, stdPhi, stdWeights);
    behavioral.setParameters(wB);
    NormalStateDependantStddevPolicy target(phi, stdPhi, stdWeights);



    // run batch training
    AdaptiveGradientStep stepl(0.1);

    OffGradType type = OffGradType::GPOMDP_BASELINE_SINGLE;
    OffPolicyGradientAlgorithm<DenseAction, DenseState> offagent(type, target, behavioral, stepl);
    auto&& batchcore = buildBatchCore(mdp, offagent);

    batchcore.getSettings().nEpisodes = 10000;
    batchcore.getSettings().episodeLength = mdp.getSettings().horizon;
    batchcore.getSettings().maxBatchIterations = 40;

    auto&& dataBegin = batchcore.runTest();
    std::cout << "initial reward: " << dataBegin.getMeanReward(mdp.getSettings().gamma) << std::endl;

    batchcore.run(behavioral);
    auto&& dataEnd = batchcore.runTest();

    std::cout << "reward end: " << dataEnd.getMeanReward(mdp.getSettings().gamma) << std::endl;


    return 0;
}
