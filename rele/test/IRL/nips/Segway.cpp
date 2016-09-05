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
#include "rele/statistics/DifferentiableNormals.h"
#include "rele/core/Core.h"
#include "rele/policy/parametric/differentiable/LinearPolicy.h"
#include "rele/approximators/features/DenseFeatures.h"
#include "rele/approximators/basis/IdentityBasis.h"
#include "rele/utils/FileManager.h"
#include "rele/environments/Segway.h"

using namespace std;
using namespace ReLe;
using namespace arma;

int main(int argc, char *argv[])
{
    //parameters
    unsigned int datasetSize = 1;

    unsigned int episodesPerPolicy = 1;
    unsigned int policyPerUpdates = 100;
    unsigned int learningEpisodes = 10000;


    //Learn optimal policy
    FileManager fm("segway", "BBO");
    fm.createDir();
    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);

    Segway mdp;

    BasisFunctions basis = IdentityBasis::generate(mdp.getSettings().stateDimensionality);
    DenseFeatures phi(basis);
    DetLinearPolicy<DenseState> policy(phi);
    int nparams = phi.rows();
    arma::vec mean(nparams, fill::ones);
    arma::mat cov(nparams, nparams, arma::fill::eye);
    cov *= 0.001;
    mean(0) = 130.1918;
    mean(1) = 51.0960;
    mean(2) = 3.0436;
    ParametricNormal expertDist(mean, cov);

    AdaptiveGradientStep stepRule(0.01);
    PGPE<DenseAction, DenseState> agent(expertDist, policy, episodesPerPolicy, policyPerUpdates,
                                        stepRule, true);

    Core<DenseAction, DenseState> core(mdp, agent);

    core.getSettings().episodeN = learningEpisodes;
    core.getSettings().episodeLength = mdp.getSettings().horizon;
    core.getSettings().testEpisodeN = datasetSize;

    std::cout << core.runEvaluation() << std::endl;

    core.runEpisodes();

    CollectorStrategy<DenseAction, DenseState> strategy;
    core.getSettings().loggerStrategy = &strategy;

    core.runTestEpisodes();

    std::cout << strategy.data.getMeanReward(mdp.getSettings().gamma) << std::endl;

    strategy.data.writeToStream(std::cout);



}
