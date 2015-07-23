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

#include "features/SparseFeatures.h"
#include "features/DenseFeatures.h"
#include "regressors/LinearApproximator.h"
#include "basis/QuadraticBasis.h"
#include "basis/IdentityBasis.h"
#include "Core.h"

#include "parametric/differentiable/NormalPolicy.h"

#include "NLS.h"

#include "PolicyEvalAgent.h"
#include "algorithms/GIRL.h"

#include "FileManager.h"

using namespace std;
using namespace arma;
using namespace ReLe;

int main(int argc, char *argv[])
{
//  RandomGenerator::seed(45423424);
//  RandomGenerator::seed(8763575);

    IRLGradType atype = IRLGradType::GB;
    vec eReward = {0.2, 0.7, 0.1};
    int nbEpisodes = 2000;

    FileManager fm("lqr", "GIRL");
    fm.createDir();
    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);

    /* Learn lqr correct policy */
    NLS mdp;

    int dim = mdp.getSettings().continuosStateDim;

    BasisFunctions basis = IdentityBasis::generate(dim);
    DenseFeatures phi(basis);

    BasisFunctions stdBasis = IdentityBasis::generate(dim);
    DenseFeatures stdPhi(stdBasis);
    arma::vec stdWeights(stdPhi.rows());
    stdWeights.fill(0.5);

    NormalStateDependantStddevPolicy expertPolicy(phi, stdPhi, stdWeights);

    /*** solve the problem in exact way ***/
    arma::vec p(2);
    p(0) = -2.8;
    p(1) = 7.3;

    expertPolicy.setParameters(p);

    std::cout << "Rewards: ";
    for (int i = 0; i < eReward.n_elem; ++i)
    {
        std::cout << eReward(i) << " ";
    }
    std::cout << "| Params: " << expertPolicy.getParameters().t() << std::endl;


    PolicyEvalAgent<DenseAction, DenseState> expert(expertPolicy);

    /* Generate LQR expert dataset */
    Core<DenseAction, DenseState> expertCore(mdp, expert);
    CollectorStrategy<DenseAction, DenseState> collection;
    expertCore.getSettings().loggerStrategy = &collection;
    expertCore.getSettings().episodeLenght = mdp.getSettings().horizon;
    expertCore.getSettings().testEpisodeN = nbEpisodes;
    expertCore.runTestEpisodes();
    Dataset<DenseAction,DenseState>& data = collection.data;


    /* Create parametric reward */
    BasisFunctions basisReward = IdentityBasis::generate(dim);
    DenseFeatures phiReward(basisReward);


    LinearApproximator rewardRegressor(phiReward);
    GIRL<DenseAction,DenseState> irlAlg(data, expertPolicy, rewardRegressor,
                                        mdp.getSettings().gamma, atype);


    //Run GIRL
    irlAlg.run();
    arma::vec gnormw = irlAlg.getWeights();

    cout << "Weights (gnorm): " << gnormw.t();

    return 0;
}
