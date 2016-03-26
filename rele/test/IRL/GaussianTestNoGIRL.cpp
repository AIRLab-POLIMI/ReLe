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
#include "rele/approximators/regressors/others/GaussianMixtureModels.h"
#include "rele/approximators/basis/PolynomialFunction.h"
#include "rele/approximators/basis/IdentityBasis.h"

#include "rele/policy/parametric/differentiable/NormalPolicy.h"

#include "rele/environments/GaussianRewardMDP.h"
#include "rele/core/Core.h"
#include "rele/core/PolicyEvalAgent.h"
#include "rele/IRL/algorithms/GIRL.h"
#include "rele/IRL/algorithms/NoGIRL.h"
#include "rele/algorithms/policy_search/gradient/GPOMDPAlgorithm.h"

#include "rele/IRL/ParametricRewardMDP.h"

#include "rele/utils/FileManager.h"

using namespace std;
using namespace arma;
using namespace ReLe;

//#define PRINT
#define TRAJECTORIES
#define RUN
#define RECOVER

int main(int argc, char *argv[])
{
//  RandomGenerator::seed(45423424);
//  RandomGenerator::seed(8763575);

    IrlGrad atype = IrlGrad::GPOMDP_BASELINE;
    int dim = 2;
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

    BasisFunctions basisStdDev = PolynomialFunction::generate(1, dim);
    SparseFeatures phiStdDev(basisStdDev, dim);
    arma::mat stdDevW(dim, phiStdDev.rows(), fill::zeros);

    for(int i = 0; i < stdDevW.n_rows; i++)
        for(int j = i*basis.size(); j < (i+1)*basis.size(); j++)
        {
            stdDevW(i, j) = 1;
        }

    stdDevW *= 0.01;

    std::cout << stdDevW << std::endl;

    MVNStateDependantStddevPolicy expertPolicy(phi, phiStdDev, stdDevW);

    /*** solve the problem ***/
    int episodesPerPolicy = 1;
    int policyPerUpdate = 100;
    int updates = 400;
    int episodes = episodesPerPolicy*policyPerUpdate*updates;
    AdaptiveGradientStep stepRule(0.01);

    GPOMDPAlgorithm<DenseAction, DenseState> expert(expertPolicy, policyPerUpdate,
            mdp.getSettings().horizon, stepRule, GPOMDPAlgorithm<DenseAction, DenseState>::BaseLineType::MULTI);

    Core<DenseAction, DenseState> expertCore(mdp, expert);
    expertCore.getSettings().loggerStrategy = nullptr;
    expertCore.getSettings().episodeLength = mdp.getSettings().horizon;
    expertCore.getSettings().episodeN = episodes;
    expertCore.getSettings().testEpisodeN = 100;
    expertCore.runEpisodes();

    cout << "Policy weights: " << expertPolicy.getParameters().t() << endl;
    cout << "Policy performaces: " << as_scalar(expertCore.runEvaluation()) << endl;


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
    arma::vec gnormw = rewardRegressor.getParameters();

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
    std::ofstream ofs(fm.addPath("Trajectories.log"));
    data.writeToStream(ofs);
#endif

#ifdef RECOVER
    ParametricRewardMDP<DenseAction, DenseState> prMDP(mdp, rewardRegressor);

    MVNStateDependantStddevPolicy imitatorPolicy(phi, phiStdDev, stdDevW);
    GPOMDPAlgorithm<DenseAction, DenseState> imitator(imitatorPolicy, policyPerUpdate,
            mdp.getSettings().horizon, stepRule, GPOMDPAlgorithm<DenseAction, DenseState>::BaseLineType::MULTI);

    Core<DenseAction, DenseState> imitatorCore(prMDP, imitator);
    imitatorCore.getSettings().loggerStrategy = nullptr;
    imitatorCore.getSettings().episodeLength = mdp.getSettings().horizon;
    imitatorCore.getSettings().episodeN = episodes;
    imitatorCore.getSettings().testEpisodeN = 100;
    imitatorCore.runEpisodes();

    cout << "Parameters learned: " << imitatorPolicy.getParameters().t() << endl;


    //Save sample trajectories
    imitatorCore.getSettings().loggerStrategy = new WriteStrategy<DenseAction, DenseState>(
        fm.addPath("Imitator.log"),  WriteStrategy<DenseAction, DenseState>::TRANS,false);
    imitatorCore.getSettings().testEpisodeN = 3000;
    imitatorCore.runTestEpisodes();
    //---

    delete imitatorCore.getSettings().loggerStrategy;

#endif
    return 0;
}
