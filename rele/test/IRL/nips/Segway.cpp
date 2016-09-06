/*
 * rele,
 *
 *
 * Copyright (C) 2016 Davide Tateo
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


#include "rele/algorithms/policy_search/PGPE/PGPE.h"

#include "rele/core/PolicyEvalAgent.h"
#include "rele/statistics/DifferentiableNormals.h"
#include "rele/core/Core.h"
#include "rele/policy/parametric/differentiable/LinearPolicy.h"
#include "rele/policy/parametric/differentiable/NormalPolicy.h"
#include "rele/approximators/features/DenseFeatures.h"
#include "rele/approximators/basis/IdentityBasis.h"
#include "rele/utils/FileManager.h"
#include "rele/environments/Segway.h"

#include "rele/IRL/ParametricRewardMDP.h"
#include "rele/IRL/algorithms/EGIRL.h"
#include "rele/IRL/algorithms/SDPEGIRL.h"
#include "rele/IRL/algorithms/CurvatureEGIRL.h"
#include "rele/IRL/algorithms/SCIRL.h"
#include "rele/IRL/algorithms/CSI.h"

#include "rele/algorithms/policy_search/gradient/REINFORCEAlgorithm.h"

#include "rele/utils/DatasetDiscretizator.h"
#include "rele/core/callbacks/CoreCallback.h"

#include "../RewardBasisLQR.h"

using namespace std;
using namespace ReLe;
using namespace arma;

#define RUN

/*AdaptiveGradientStep stepRule(0.01);
PGPE<DenseAction, DenseState> agent(expertDist, policy, episodesPerPolicy, policyPerUpdates,
                                    stepRule, true);*/

arma::vec learnSegway(Environment<DenseAction, DenseState>& mdp, DenseFeatures& phi)
{
    //Learning parameters
    int episodesPerPolicy = 1;
    int policyPerUpdate = 100;
    int updates = 400;
    int episodes = episodesPerPolicy*policyPerUpdate*updates;
    int testEpisodes = 1000;
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
    //parameters
    unsigned int datasetSize = 10;
    std::string n_discretizations("5");


    unsigned int episodesPerPolicy = 1;
    unsigned int policyPerUpdates = 100;
    unsigned int learningEpisodes = 10000;

    FileManager fm("nips/segway");///"+n_episodes+"/"+n_experiment);
    fm.cleanDir();
    fm.createDir();
    std::cout << std::setprecision(OS_PRECISION);

    Segway mdp;

    //Create expert policy
    BasisFunctions basis = IdentityBasis::generate(mdp.getSettings().stateDimensionality);
    DenseFeatures phi(basis);
    DetLinearPolicy<DenseState> policy(phi);
    int nparams = phi.rows();
    arma::vec mean(nparams, fill::ones);
    arma::mat cov(nparams, nparams, arma::fill::eye);
    cov *= 0.001;
    mean(0) = 1.2430e+02;
    mean(1) = 5.5160e+01;
    mean(2) = 1.0500e-01;
    ParametricNormal expertDist(mean, cov);

    //Create expert dataset
    PolicyEvalDistribution<DenseAction, DenseState> expert(expertDist, policy);
    CollectorStrategy<DenseAction, DenseState> strategy;

    auto&& core = buildCore(mdp, expert);

    core.getSettings().episodeN = learningEpisodes;
    core.getSettings().episodeLength = mdp.getSettings().horizon;
    core.getSettings().testEpisodeN = datasetSize;
    core.getSettings().loggerStrategy = &strategy;

    core.runTestEpisodes();

    Dataset<DenseAction, DenseState>& data = strategy.data;

    arma::mat theta = expert.getParams();

    ofstream ofs1(fm.addPath("TrajectoriesExpert.txt"));
    data.writeToStream(ofs1);

#ifdef RUN
    // -- RUN -- //

    // Create parametric reward
    BasisFunctions basisReward;
    for(unsigned int i = 0; i < 3; i++)
        basisReward.push_back(new LQR_RewardBasis(i, 3));
    DenseFeatures phiReward(basisReward);
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
    unsigned int discretizedActions = std::stoi(n_discretizations);

    //FIXME
    IRLAlgorithm<FiniteAction, DenseState>* irlAlg_c[2];
    /*BasisFunctions basis_c = GaussianRbf::generate(
    {
        3,
        3,
        6,
        2,
        discretizedActions
    },
    {
        0.0, 150.0,
        0.0, 150.0,
        -M_PI, M_PI,
        -15.0, 15.0,
        -0.5, discretizedActions+0.5
    });

    DenseFeatures phi_c(basis_c);


    DatasetDiscretizator discretizator(Range(-15, 15), discretizedActions);
    auto&& discretizedData = discretizator.discretize(data);

    irlAlg_c[0] = new SCIRL<DenseState>(discretizedData,rewardRegressor, mdp.getSettings().gamma, discretizedActions);
    irlAlg_c[1] = new CSI<DenseState>(discretizedData, phi_c, rewardRegressor, mdp.getSettings().gamma, discretizedActions);

    for(unsigned int i = 0; i < 2; i++)
    {
        irlAlg_c[i]->run();
        weights.col(numAlg+i) = rewardRegressor.getParameters();
        delete irlAlg_c[i];
    }*/

    // -- Save weights -- //
    weights.save(fm.addPath("Weights.txt"), arma::raw_ascii);


    // -- RECOVER! -- //
    for(unsigned int i = 0; i < numAlg /*+ 2*/; i++)//FIXME
    {
        rewardRegressor.setParameters(weights.col(i));
        ParametricRewardMDP<DenseAction, DenseState> prMdp(mdp, rewardRegressor);
        arma::vec mean = learnSegway(prMdp, phi);;

        double epsilon = 0.05;
        NormalPolicy imitatorPolicy(epsilon, phi);
        imitatorPolicy.setParameters(mean);

        //Evaluate policy against the real mdp
        PolicyEvalAgent<DenseAction, DenseState> imitator(imitatorPolicy);
        Core<DenseAction, DenseState> evaluationCore(mdp, imitator);
        CollectorStrategy<DenseAction, DenseState> collector2;
        evaluationCore.getSettings().loggerStrategy = &collector2;
        evaluationCore.getSettings().episodeLength = mdp.getSettings().horizon;
        evaluationCore.getSettings().testEpisodeN = 100;

        evaluationCore.runTestEpisodes();

        Dataset<DenseAction,DenseState>& data2 = collector2.data;

        double gamma = mdp.getSettings().gamma;

        arma::vec reward = data2.getMeanReward(mdp.getSettings().gamma);
        cout << "reward: " << arma::as_scalar(reward) << endl;

        arma::vec featuresExpectation = data2.computefeatureExpectation(phiReward, mdp.getSettings().gamma);

        reward.save(fm.addPath("reward_" + std::to_string(i) + ".txt"), arma::raw_ascii);
        featuresExpectation.save(fm.addPath("featuresExpectation_" + std::to_string(i) + ".txt"), arma::raw_ascii);

        ofstream ofs2(fm.addPath("TrajectoriesImitator_" + std::to_string(i) +".txt"));
        data2.writeToStream(ofs2);
    }

    arma::vec featuresExpectationExpert = data.computefeatureExpectation(phiReward, mdp.getSettings().gamma);
    featuresExpectationExpert.save(fm.addPath("featuresExpectationExpert.txt"), arma::raw_ascii);

#endif


}
