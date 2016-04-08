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

#include "rele/core/BatchCore.h"
#include "rele/policy/parametric/differentiable/PortfolioNormalPolicy.h"
#include "rele/algorithms/batch/policy_search/gradient/OffPolicyGradientAlgorithm.h"
#include "rele/algorithms/batch/policy_search/gradient/MBPGA.h"
#include "rele/approximators/basis/IdentityBasis.h"
#include "rele/approximators/features/DenseFeatures.h"
#include "rele/utils/RandomGenerator.h"
#include "rele/utils/FileManager.h"
#include "rele/environments/Portfolio.h"

using namespace std;
using namespace ReLe;
using namespace arma;

int main()
{
    Portfolio mdp;

    int dim = mdp.getSettings().stateDimensionality;

    BasisFunctions basis = IdentityBasis::generate(dim);
    DenseFeatures phi(basis);

    double epsilon = 0.05;
    PortfolioNormalPolicy target(epsilon, phi);
    PortfolioNormalPolicy behavioural(epsilon, phi);

    // run batch training
    AdaptiveGradientStep stepl(0.1);
    IndexRT rewardF(0);

    OffGradType type = OffGradType::GPOMDP_BASELINE_SINGLE;
    //OffPolicyGradientAlgorithm<FiniteAction, DenseState> offagent(type, target, behavioural, stepl, &rewardF);

    double penalization = 0.5;
    MBPGA<FiniteAction, DenseState> offagent(target, behavioural, stepl, &rewardF, penalization);
    auto&& batchcore = buildBatchCore(mdp, offagent);

    batchcore.getSettings().nEpisodes = 10000;
    batchcore.getSettings().episodeLength = mdp.getSettings().horizon;
    batchcore.getSettings().maxBatchIterations = 40;

    auto&& dataBegin = batchcore.runTest();
    std::cout << "initial reward: " << dataBegin.getMeanReward(mdp.getSettings().gamma) << std::endl;

    batchcore.run(behavioural);
    auto&& dataEnd = batchcore.runTest();

    std::cout << "reward end: " << dataEnd.getMeanReward(mdp.getSettings().gamma) << std::endl;


}
