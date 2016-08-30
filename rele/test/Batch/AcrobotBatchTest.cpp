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

#include "rele/core/BatchCore.h"
#include "rele/algorithms/batch/td/FQI.h"
#include "rele/approximators/regressors/trees/ExtraTree.h"
#include "rele/environments/Acrobot.h"
#include "rele/policy/q_policy/e_Greedy.h"
#include "rele/approximators/regressors/trees/ExtraTreeEnsemble.h"
#include "rele/approximators/basis/IdentityBasis.h"
#include "rele/approximators/features/DenseFeatures.h"
#include "rele/utils/FileManager.h"

#include <string>

using namespace std;
using namespace ReLe;

int main(int argc, char *argv[])
{
    FileManager fm("acro", "fqi");
    fm.createDir();
    fm.cleanDir();

    // Define domain
    Acrobot mdp;

    BasisFunctions bfs;
    bfs = IdentityBasis::generate(mdp.getSettings().stateDimensionality + mdp.getSettings().actionDimensionality);

    DenseFeatures phi(bfs);

    // Define tree regressors
    arma::vec defaultValue = {0};
    EmptyTreeNode<arma::vec> defaultNode(defaultValue);
    ExtraTreeEnsemble_<arma::vec, arma::vec> QRegressor(phi, defaultNode);

    // Define algorithm
    double epsilon = 1e-6;
    BatchTDAgent<DenseState>* batchAgent = new FQI(QRegressor, epsilon);

    e_GreedyApproximate policy;
    policy.setNactions(mdp.getSettings().actionsNumber);

    //Run experiments and learning
    batchAgent->setPolicy(policy);
    auto&& core = buildBatchCore(mdp, *batchAgent);

    core.getSettings().episodeLength = mdp.getSettings().horizon;
    core.getSettings().nEpisodes = 100;
    core.getSettings().maxBatchIterations = 100;
    core.getSettings().datasetLogger = new WriteBatchDatasetLogger<FiniteAction, DenseState>(fm.addPath("acro.log"));
    core.getSettings().agentLogger = new BatchAgentPrintLogger<FiniteAction, DenseState>();

    core.run(20);

    return 0;
}
