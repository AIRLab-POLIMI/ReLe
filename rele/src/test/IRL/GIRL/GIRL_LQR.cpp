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
#include "basis/IdentityBasis.h"

#include "parametric/differentiable/NormalPolicy.h"

#include "LQR.h"
#include "LQRsolver.h"
#include "PolicyEvalAgent.h"
#include "algorithms/GIRL.h"
#include "algorithms/PGIRL.h"

#include "FileManager.h"

#include "RewardBasisLQR.h"

using namespace std;
using namespace arma;
using namespace ReLe;

#define RUN
//#define PRINT

int main(int argc, char *argv[])
{
//  RandomGenerator::seed(45423424);
//  RandomGenerator::seed(8763575);

    IRLGradType atype = IRLGradType::GB;
#ifndef PRINT
    vec eReward = {0.2, 0.7, 0.1};
#else
    vec eReward = {0.3, 0.7};
#endif
    int nbEpisodes = 5000;

    FileManager fm("lqr", "GIRL");
    fm.createDir();
    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);

    /* Learn lqr correct policy */
    int dim = eReward.n_elem;
    LQR mdp(dim, dim);

    BasisFunctions basis;
    for (int i = 0; i < dim; ++i)
    {
        basis.push_back(new IdentityBasis(i));
    }

    SparseFeatures phi;
    phi.setDiagonal(basis);

    MVNPolicy expertPolicy(phi);

    /*** solve the problem in exact way ***/
    LQRsolver solver(mdp,phi);
    solver.setRewardWeights(eReward);
    mat K = solver.computeOptSolution();
    arma::vec p = K.diag();
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
    BasisFunctions basisReward;
    for(unsigned int i = 0; i < eReward.n_elem; i++)
        basisReward.push_back(new LQR_RewardBasis(i, dim));
    DenseFeatures phiReward(basisReward);


    LinearApproximator rewardRegressor(phiReward);
    GIRL<DenseAction,DenseState> irlAlg(data, expertPolicy, rewardRegressor,
                                        mdp.getSettings().gamma, atype);

    PlaneGIRL<DenseAction, DenseState> irlAlg2(data, expertPolicy, basisReward,
            mdp.getSettings().gamma, atype);

#ifdef RUN
    //Run GIRL
    irlAlg.run();
    arma::vec gnormw = irlAlg.getWeights();

    //Run PGIRL
    irlAlg2.run();
    arma::vec planew = irlAlg2.getWeights();


    //Print results
    cout << "Weights (gnorm): " << gnormw.t();
    cout << "Weights (plane): " << planew.t();
#endif

#ifdef PRINT
    //calculate full grid function
    int samplesParams = 101;
    arma::vec valuesG(samplesParams);
    arma::vec valuesJ(samplesParams);
    arma::vec valuesD(samplesParams);
    arma::mat valuesdG2(dim, samplesParams);

    for(int i = 0; i < samplesParams; i++)
    {
        cerr << i << endl;
        double step = 0.01;
        arma::vec wm(2);
        wm(0) = i*step;
        wm(1) = 1.0 - wm(0);
        rewardRegressor.setParameters(wm);
        arma::mat dGradient(dim, dim);
        arma::vec dJ;
        arma::vec g = irlAlg.ReinforceBaseGradient(dGradient);
        arma::vec dg2 = 2.0*dGradient.t() * g;

        double Je = irlAlg.computeJ(dJ);
        double D = irlAlg.computeDisparity();
        double G = norm(g);
        valuesG(i) = G;
        valuesJ(i) = Je;
        valuesD(i) = D;
        valuesdG2.col(i) = dg2;
    }

    valuesG.save("/tmp/ReLe/G.txt", arma::raw_ascii);
    valuesJ.save("/tmp/ReLe/J.txt", arma::raw_ascii);
    valuesD.save("/tmp/ReLe/D.txt", arma::raw_ascii);
    valuesdG2.save("/tmp/ReLe/dG2.txt", arma::raw_ascii);
#endif

    return 0;
}
