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

#include "features/DenseFeatures.h"
#include "regressors/GaussianMixtureModels.h"
#include "parametric/differentiable/LinearPolicy.h"
#include "basis/IdentityBasis.h"

#include "DifferentiableNormals.h"

#include "NLS.h"

#include "PolicyEvalAgent.h"
#include "Core.h"

#include "FileManager.h"

using namespace std;
using namespace arma;
using namespace ReLe;

int main(int argc, char *argv[])
{
//  RandomGenerator::seed(45423424);
//  RandomGenerator::seed(8763575);

    int nbEpisodes = 3000;

    FileManager fm("nls", "GMM");
    fm.createDir();
    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);


    NLS mdp;

    //Setup expert policy
    int dim = mdp.getSettings().continuosStateDim;

    BasisFunctions basis = IdentityBasis::generate(dim);
    DenseFeatures phi(basis);

    arma::vec p(2);
    p(0) = 6.5178;
    p(1) = -2.5994;

    DetLinearPolicy<DenseState> expertPolicy(phi);
    ParametricFullNormal expertDist(p, 0.1*arma::eye(p.size(), p.size()));

    std::cout << "Params: " << expertDist.getParameters().t() << std::endl;

    PolicyEvalDistribution<DenseAction, DenseState> expert(expertDist, expertPolicy);

    /* Generate LQR expert dataset */
    Core<DenseAction, DenseState> expertCore(mdp, expert);
    CollectorStrategy<DenseAction, DenseState> collection;
    expertCore.getSettings().loggerStrategy = &collection;
    expertCore.getSettings().episodeLength = mdp.getSettings().horizon;
    expertCore.getSettings().testEpisodeN = nbEpisodes;
    expertCore.runTestEpisodes();
    Dataset<DenseAction,DenseState>& data = collection.data;

    BasisFunctions basisGMM = IdentityBasis::generate(3);
    DenseFeatures phiGMM(basisGMM);
    GaussianMixtureRegressor featureRegressor(phiGMM, 10, 10);

    arma::mat featuresExp = data.computeEpisodeFeatureExpectation(phiGMM, mdp.getSettings().gamma);

    featuresExp.save(fm.addPath("Phi.txt"),  arma::raw_ascii);

    featureRegressor.trainFeatures(featuresExp);

    for(unsigned int i = 0; i < featureRegressor.getCurrentK(); i++)
        std::cout << "mu[" << i << "] = " << featureRegressor.getMu(i) << std::endl;

}
