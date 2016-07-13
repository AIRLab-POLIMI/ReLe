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

#include "rele/approximators/features/SparseFeatures.h"
#include "rele/approximators/features/DenseFeatures.h"
#include "rele/approximators/features/TilesCoder.h"
#include "rele/approximators/regressors/others/GaussianMixtureModels.h"
#include "rele/approximators/basis/IdentityBasis.h"
#include "rele/approximators/basis/PolynomialFunction.h"
#include "rele/approximators/basis/GaussianRbf.h"
#include "rele/approximators/tiles/BasicTiles.h"

#include "rele/policy/parametric/differentiable/NormalPolicy.h"
#include "rele/policy/parametric/differentiable/LinearPolicy.h"
#include "rele/statistics/DifferentiableNormals.h"

#include "rele/environments/ShipSteering.h"

#include "rele/core/PolicyEvalAgent.h"
#include "rele/core/Core.h"

#include "rele/IRL/ParametricRewardMDP.h"
#include "rele/IRL/algorithms/EGIRL.h"
#include "rele/IRL/algorithms/SDPEGIRL.h"
#include "rele/IRL/algorithms/CurvatureEGIRL.h"
#include "rele/IRL/algorithms/SCIRL.h"
#include "rele/IRL/algorithms/CSI.h"

#include "rele/algorithms/policy_search/gradient/REINFORCEAlgorithm.h"

#include "rele/utils/FileManager.h"
#include "rele/utils/DatasetDiscretizator.h"
#include "rele/core/callbacks/CoreCallback.h"

using namespace std;
using namespace arma;
using namespace ReLe;

#define RUN

arma::vec learnShipSteering(Environment<DenseAction, DenseState>& mdp, DenseFeatures& phi, int nbEpisodes)
{
    //Learning parameters
    int episodesPerPolicy = 1;
    int policyPerUpdate = 100;
    int updates = 400;
    int episodes = episodesPerPolicy*policyPerUpdate*updates;
    int testEpisodes = 100;
    AdaptiveGradientStep stepRule(0.01);

    int dim = mdp.getSettings().stateDimensionality;

    double epsilon = 0.05;
    NormalPolicy policy(epsilon, phi);

    // Solve the problem with REINFORCE
    REINFORCEAlgorithm<DenseAction, DenseState> expert(policy, policyPerUpdate, stepRule);
    CoreProgressBar progressBar;

    Core<DenseAction, DenseState> expertCore(mdp, expert);
    expertCore.getSettings().loggerStrategy = nullptr;
    expertCore.getSettings().episodeLength = mdp.getSettings().horizon;
    expertCore.getSettings().episodeN = episodes;
    expertCore.getSettings().testEpisodeN = testEpisodes;
    expertCore.getSettings().episodeCallback = &progressBar;
    expertCore.runEpisodes();

    return policy.getParameters();
}

int main(int argc, char *argv[])
{
    if(argc != 3)
    {
        cout << "Wrong argument number: n_episode, n_experiment must be provided" << endl;
        return -1;
    }

    string n_episodes(argv[1]);
    string n_experiment(argv[2]);

    FileManager fm("nips/ship/"+n_episodes+"/"+n_experiment);
    fm.cleanDir();
    fm.createDir();
    std::cout << std::setprecision(OS_PRECISION);

    unsigned int nbEpisodes = std::stod(n_episodes);


    ShipSteering mdp;

    //Setup expert policy
    int dim = mdp.getSettings().stateDimensionality;

    BasisFunctions basis = GaussianRbf::generate(
    {
        3,
        3,
        6,
        2
    },
    {
        0.0, 150.0,
        0.0, 150.0,
        -M_PI, M_PI,
        -15.0, 15.0
    });

    DenseFeatures phi(basis);

#ifdef LEARN
    arma::vec p = learnShipSteering(mdp, phi, 1000);
    p.save("/tmp/ReLe/nips/ship/ExpertParams.txt",  arma::raw_ascii);
#else
    arma::vec p;
    p.load("/tmp/ReLe/nips/ship/ExpertParams.txt", arma::raw_ascii);
#endif

    DetLinearPolicy<DenseState> expertPolicy(phi);
    ParametricNormal expertDist(p, 0.1*arma::eye(p.size(), p.size()));

    PolicyEvalDistribution<DenseAction, DenseState> expert(expertDist, expertPolicy);

    // -- Generate expert dataset -- //
    Core<DenseAction, DenseState> expertCore(mdp, expert);
    CollectorStrategy<DenseAction, DenseState> collection;
    expertCore.getSettings().loggerStrategy = &collection;
    expertCore.getSettings().episodeLength = mdp.getSettings().horizon;
    expertCore.getSettings().testEpisodeN = nbEpisodes;
    expertCore.runTestEpisodes();
    Dataset<DenseAction,DenseState>& data = collection.data;


    arma::mat theta = expert.getParams();

    ofstream ofs1(fm.addPath("TrajectoriesExpert.txt"));
    data.writeToStream(ofs1);

#ifdef RUN
    // -- RUN -- //

    // Create parametric reward
    //BasisFunctions basisReward = GaussianRbf::generate({20, 20}, {0, 150, 0, 150});
    //DenseFeatures phiReward(basisReward);

    auto* tiles = new BasicTiles({Range(0, 150), Range(0, 150)}, {20, 20});
    DenseTilesCoder phiReward(tiles);

    LinearApproximator rewardRegressor(phiReward);


    unsigned int numAlg = 3;
    IRLAlgorithm<DenseAction, DenseState>* irlAlg[numAlg];


    irlAlg[0] = new EGIRL<DenseAction,DenseState>(data, theta, expertDist, rewardRegressor, mdp.getSettings().gamma, IrlEpGrad::PGPE_BASELINE);
    irlAlg[1] = new SDPEGIRL<DenseAction,DenseState>(data, theta, expertDist, rewardRegressor, mdp.getSettings().gamma, IrlEpGrad::PGPE_BASELINE, IrlEpHess::PGPE_BASELINE);
    irlAlg[2] = new CurvatureEGIRL<DenseAction,DenseState>(data, theta, expertDist, rewardRegressor, mdp.getSettings().gamma, IrlEpGrad::PGPE_BASELINE, IrlEpHess::PGPE_BASELINE, 1e-4);

    arma::mat weights(rewardRegressor.getParametersSize(), numAlg + 2, arma::fill::zeros);

    for(unsigned int i = 0; i < numAlg; i++)
    {
        irlAlg[i]->run();
        weights.col(i) = rewardRegressor.getParameters();
        delete irlAlg[i];
    }

    //Classification Based
    IRLAlgorithm<FiniteAction, DenseState>* irlAlg_c[2];
    BasisFunctions basis_c = GaussianRbf::generate(
    {
        3,
        3,
        6,
        2,
        30
    },
    {
        0.0, 150.0,
        0.0, 150.0,
        -M_PI, M_PI,
        -15.0, 15.0,
        -0.5, 30.5
    });

    DenseFeatures phi_c(basis_c);

    unsigned int discretizedActions = 30;
    DatasetDiscretizator discretizator(Range(-15, 15), discretizedActions);
    auto&& discretizedData = discretizator.discretize(data);

    irlAlg_c[0] = new SCIRL<DenseState>(discretizedData,rewardRegressor, mdp.getSettings().gamma, discretizedActions);
    irlAlg_c[1] = new CSI<DenseState>(discretizedData, phi_c, rewardRegressor, mdp.getSettings().gamma, discretizedActions);

    for(unsigned int i = 0; i < 2; i++)
    {
        irlAlg_c[i]->run();
        weights.col(numAlg+i) = rewardRegressor.getParameters();
        delete irlAlg_c[i];
    }

    // -- Save weights -- //
    weights.save(fm.addPath("Weights.txt"), arma::raw_ascii);


    // -- RECOVER! -- //

    //Try to recover the initial policy
    int episodesPerPolicy = 1;
    int policyPerUpdate = 100;
    int updates = 400;
    int episodes = episodesPerPolicy*policyPerUpdate*updates;


    BasisFunctions stdBasis = PolynomialFunction::generate(1, dim);
    DenseFeatures stdPhi(stdBasis);
    arma::vec stdWeights(stdPhi.rows());
    stdWeights.fill(0.1);

    for(unsigned int i = 0; i < numAlg + 2; i++)
    {
        rewardRegressor.setParameters(weights.col(i));
        ParametricRewardMDP<DenseAction, DenseState> prMdp(mdp, rewardRegressor);
        arma::vec mean = learnShipSteering(prMdp, phi, nbEpisodes);;

        double epsilon = 0.05;
        NormalPolicy imitatorPolicy(epsilon, phi);
        imitatorPolicy.setParameters(mean);

        //Evaluate policy against the real mdp
        PolicyEvalAgent<DenseAction, DenseState> imitator(imitatorPolicy);
        Core<DenseAction, DenseState> evaluationCore(mdp, imitator);
        CollectorStrategy<DenseAction, DenseState> collector2;
        evaluationCore.getSettings().loggerStrategy = &collector2;
        evaluationCore.getSettings().episodeLength = mdp.getSettings().horizon;
        evaluationCore.getSettings().episodeN = episodes;
        evaluationCore.getSettings().testEpisodeN = nbEpisodes;

        evaluationCore.runTestEpisodes();

        Dataset<DenseAction,DenseState>& data2 = collector2.data;

        double gamma = mdp.getSettings().gamma;

        arma::vec reward = data2.getMeanReward(mdp.getSettings().gamma);
        cout << "reward: " << arma::as_scalar(reward) << endl;

        arma::vec featuresExpectation = data2.computefeatureExpectation(phiReward, mdp.getSettings().gamma);

        reward.save(fm.addPath("reward._" + std::to_string(i) + "txt"), arma::raw_ascii);
        featuresExpectation.save(fm.addPath("featuresExpectation_" + std::to_string(i) + ".txt"), arma::raw_ascii);

        ofstream ofs2(fm.addPath("TrajectoriesImitator_" + std::to_string(i) +".txt"));
        data2.writeToStream(ofs2);
    }

#endif

    return 0;
}
