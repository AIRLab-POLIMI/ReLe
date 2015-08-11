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
#include "basis/PolynomialFunction.h"
#include "basis/IdentityBasis.h"

#include "parametric/differentiable/NormalPolicy.h"

#include "GaussianRewardMDP.h"
#include "LQRsolver.h"
#include "PolicyEvalAgent.h"
#include "algorithms/GIRL.h"
#include "algorithms/NoGIRL.h"
#include "policy_search/gradient/onpolicy/GPOMDPAlgorithm.h"

#include "ParametricRewardMDP.h"

#include "FileManager.h"

using namespace std;
using namespace arma;
using namespace ReLe;

#define PRINT
#define TRAJECTORIES
//#define RUN
//#define RECOVER

int main(int argc, char *argv[])
{
//  RandomGenerator::seed(45423424);
//  RandomGenerator::seed(8763575);

    IRLGradType atype = IRLGradType::GB;
    int dim = 1;
    int nbEpisodes = 3000;

    FileManager fm("gaussian", "GIRL");
    fm.createDir();
    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);

    /*** Learn correct policy ***/
    GaussianRewardMDP mdp(dim);

    BasisFunctions basis = PolynomialFunction::generate(1, dim);
    //BasisFunctions basis = IdentityBasis::generate(dim);

    SparseFeatures phi(basis, dim);

    arma::mat Sigma(dim, dim, fill::eye);
    Sigma *= 0.1;

    BasisFunctions basisStdDev = PolynomialFunction::generate(1, dim);
    SparseFeatures phiStdDev(basisStdDev, dim);
    arma::mat stdDevW(dim, phiStdDev.rows(), fill::ones);
    stdDevW *= 0.01;

    MVNStateDependantStddevPolicy expertPolicy(phi, phiStdDev, stdDevW);

    /*** solve the problem ***/
    int episodesPerPolicy = 1;
    int policyPerUpdate = 100;
    int updates = 400;
    int episodes = episodesPerPolicy*policyPerUpdate*updates;
    AdaptiveStep stepRule(0.01);

    GPOMDPAlgorithm<DenseAction, DenseState> expert(expertPolicy, policyPerUpdate,
            mdp.getSettings().horizon, stepRule, GPOMDPAlgorithm<DenseAction, DenseState>::BaseLineType::MULTI);

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
    std::vector<double> lowerBounds(rewardRegressor.getParametersSize(), -10.0);
    std::vector<double> upperBounds(rewardRegressor.getParametersSize(), 10.0);
    NoGIRL<DenseAction,DenseState> irlAlg(data, expertPolicy, rewardRegressor,
                                          mdp.getSettings().gamma, atype, lowerBounds, upperBounds);

#ifdef RUN
    //Run GIRL
    irlAlg.run();
    arma::vec gnormw = irlAlg.getWeights();

    //Print results
    cout << "Optimal Weights: " << arma::zeros(dim).t() << endl;
    cout << "Weights (gnorm): " << gnormw.t();
#endif

#ifdef PRINT
    //calculate full grid function
    int samplesParams = 501;
    double startValue = -5.0;
    double step = 0.02;

    arma::vec valuesG(samplesParams);
    arma::vec valuesJ(samplesParams);
    arma::vec valuesD(samplesParams);

    for(int i = 0; i < samplesParams; i++)
    {
        cout << i << "/"<< samplesParams -1 << endl;
        arma::vec wm(1);
        wm(0) = i*step + startValue;
        rewardRegressor.setParameters(wm);
        arma::mat dG2;
        arma::vec dJ;
        arma::vec dD;
        arma::vec g;
        switch(atype)
        {
        case GB:
            g = irlAlg.GpomdpBaseGradient(dG2);
            break;
        case RB:
            g = irlAlg.ReinforceBaseGradient(dG2);
            break;
        case ENAC:
            g = irlAlg.ENACGradient(dG2);
            break;
        }

        double Je = irlAlg.computeJ(dJ);
        double G2 = as_scalar(g.t()*g);
        double D = irlAlg.computeDisparity(dD);
        valuesG(i) = std::sqrt(G2);
        valuesJ(i) = Je;
        valuesD(i) = D;
    }

    valuesG.save("/tmp/ReLe/G.txt", arma::raw_ascii);
    valuesJ.save("/tmp/ReLe/J.txt", arma::raw_ascii);
    valuesD.save("/tmp/ReLe/D.txt", arma::raw_ascii);

#endif

#ifdef TRAJECTORIES
    std::ofstream ofs("/tmp/ReLe/Trajectories.txt");
    data.writeToStream(ofs);
#endif

#ifdef RECOVER
    ParametricRewardMDP<DenseAction, DenseState> prMDP(mdp, rewardRegressor);

    MVNStateDependantStddevPolicy imitatorPolicy(phi, phiStdDev, stdDevW);
    GPOMDPAlgorithm<DenseAction, DenseState> imitator(imitatorPolicy, policyPerUpdate,
            mdp.getSettings().horizon, stepRule, GPOMDPAlgorithm<DenseAction, DenseState>::BaseLineType::MULTI);

    Core<DenseAction, DenseState> imitatorCore(prMDP, imitator);
    imitatorCore.getSettings().loggerStrategy = &emptyS;
    imitatorCore.getSettings().episodeLenght = mdp.getSettings().horizon;
    imitatorCore.getSettings().episodeN = episodes;
    imitatorCore.getSettings().testEpisodeN = 100;
    imitatorCore.runEpisodes();

    cout << "Parameters learned: " << imitatorPolicy.getParameters().t() << endl;

#endif
    return 0;
}
