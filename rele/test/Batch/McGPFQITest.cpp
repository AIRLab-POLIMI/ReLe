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
#include "rele/core/PolicyEvalAgent.h"
#include "rele/policy/q_policy/e_Greedy.h"
#include "rele/algorithms/batch/td/DoubleFQI.h"
#include "rele/algorithms/batch/td/W-FQI.h"
#include "rele/approximators/features/DenseFeatures.h"
#include "rele/environments/MountainCar.h"
#include "rele/approximators/basis/IdentityBasis.h"
#include "rele/approximators/regressors/others/GaussianProcess.h"
#include "rele/approximators/regressors/others/LinearApproximator.h"
#include "rele/utils/FileManager.h"

#include <iostream>

using namespace std;
using namespace ReLe;
using namespace arma;

int main(int argc, char *argv[])
{
    MountainCar mdp;

    /*arma::vec defaultValue = {0};
    EmptyTreeNode<arma::vec> defaultNode(defaultValue);
    KDTree<arma::vec, arma::vec> QRegressorA(phi, defaultNode, 1, 1);
    KDTree<arma::vec, arma::vec> QRegressorB(phi, defaultNode, 1, 1);

    FQI<FiniteState> batchAgent(QRegressorA, nActions, epsilon);
    DoubleFQI<FiniteState> batchAgent(QRegressorA, QRegressorB, nActions, epsilon);*/

    BasisFunctions bfs;
    bfs = IdentityBasis::generate(mdp.getSettings().statesNumber + mdp.getSettings().actionsNumber);

    DenseFeatures phi(bfs);

    double epsilon = 1e-8;
    GaussianProcess QRegressor(phi);
    W_FQI batchAgent(QRegressor, mdp.getSettings().actionsNumber, epsilon);

    std::string fileName = "mc.log";
    FileManager fm("mc", "linearSarsa");
    ifstream is(fm.addPath(fileName));
    Dataset<FiniteAction, DenseState> data;
    data.readFromStream(is);
    is.close();

    auto&& core = buildBatchOnlyCore(data, batchAgent);
    core.getSettings().maxBatchIterations = 1;

    core.run(mdp.getSettings());

    e_GreedyApproximate* epsP;
    batchAgent.setPolicy(epsP);
    Policy<FiniteAction, DenseState>* policy = batchAgent.getPolicy();
    PolicyEvalAgent<FiniteAction, DenseState> testAgent(*policy);

    return 0;
}
