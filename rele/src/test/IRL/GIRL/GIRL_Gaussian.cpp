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
#include "regressors/GaussianMixtureModels.h"
#include "basis/QuadraticBasis.h"
#include "basis/IdentityBasis.h"

#include "parametric/differentiable/NormalPolicy.h"

#include "GaussianRewardMDP.h"
#include "LQRsolver.h"
#include "PolicyEvalAgent.h"
#include "algorithms/GIRL.h"
#include "policy_search/gradient/onpolicy/GPOMDPAlgorithm.h"
#include "policy_search/gradient/onpolicy/ENACAlgorithm.h"

#include "FileManager.h"

using namespace std;
using namespace arma;
using namespace ReLe;


int main(int argc, char *argv[])
{
//  RandomGenerator::seed(45423424);
//  RandomGenerator::seed(8763575);

    IRLGradType atype = IRLGradType::GB;
    int dim = 1;
    int nbEpisodes = 2000;

    FileManager fm("gaussian", "GIRL");
    fm.createDir();
    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);

    /*** Learn lqr correct policy ***/
    GaussianRewardMDP mdp(dim, 0, 0.1);

    BasisFunctions basis = IdentityBasis::generate(dim);

    SparseFeatures phi;
    phi.setDiagonal(basis);

    MVNPolicy expertPolicy(phi);

    /*** solve the problem ***/
    int episodesPerPolicy = 1;
    int policyPerUpdate = 100;
    int updates = 400;
    int episodes = episodesPerPolicy*policyPerUpdate*updates;
    AdaptiveStep stepRule(0.01);

    GPOMDPAlgorithm<DenseAction, DenseState> expert(expertPolicy, policyPerUpdate,
            mdp.getSettings().horizon, stepRule, GPOMDPAlgorithm<DenseAction, DenseState>::BaseLineType::MULTI);
    /*eNACAlgorithm<DenseAction, DenseState> expert(expertPolicy, policyPerUpdate, stepRule);*/

    Core<DenseAction, DenseState> expertCore(mdp, expert);
    EmptyStrategy<DenseAction, DenseState> emptyS;
    expertCore.getSettings().loggerStrategy = &emptyS;
    expertCore.getSettings().episodeLenght = mdp.getSettings().horizon;
    expertCore.getSettings().episodeN = episodes;
    expertCore.getSettings().testEpisodeN = 100;
    expertCore.runEpisodes();

    cout << "Policy weights: " << expertPolicy.getParameters().t() << endl;
    cout << "Policy performaces: " << as_scalar(expertCore.runBatchTest()) << endl;


    /*** Collect expert data ***/
    CollectorStrategy<DenseAction, DenseState> collection;
    expertCore.getSettings().loggerStrategy = &collection;
    expertCore.getSettings().testEpisodeN = nbEpisodes;
    expertCore.runTestEpisodes();
    Dataset<DenseAction,DenseState>& data = collection.data;


    /*** Create parametric reward ***/
    BasisFunctions basisReward;
    for(unsigned int i = 0; i < dim; i++)
        basisReward.push_back(new IdentityBasis(2*dim+i));
    DenseFeatures phiReward(basisReward);


    GaussianRegressor rewardRegressor(phiReward);
    GIRL<DenseAction,DenseState> irlAlg(data, expertPolicy, rewardRegressor,
                                        mdp.getSettings().gamma, atype, false);


    //Run GIRL
    //irlAlg.run();
    //arma::vec gnormw = irlAlg.getWeights();

    //calculate full grid function
    int samplesPositive = 1000;
    arma::vec values(samplesPositive*2+1);
    arma::mat gradients(samplesPositive*2+1, rewardRegressor.getParametersSize());
    for(int i = -samplesPositive; i < samplesPositive+1; i++)
    {
    	double step = 0.01;
    	arma::vec wm(1);
    	wm(0) = i*step;
    	rewardRegressor.setParameters(wm);
    	arma::mat gGrad(dim, dim);
    	arma::vec grad = irlAlg.ReinforceGradient(gGrad);
    	gradients.row(i+samplesPositive) = gGrad.t()*grad;
    	values(i+samplesPositive) = as_scalar(0.5*grad.t()*grad);
    	cout << i << endl;
    }

    values.save("/tmp/ReLe/norm.txt", arma::raw_ascii);
    gradients.save("/tmp/ReLe/gradient.txt", arma::raw_ascii);


    //Print results
    cout << "Optimal Weights: " << arma::zeros(dim).t() << endl;
    //cout << "Weights (gnorm): " << gnormw.t();

    return 0;
}
