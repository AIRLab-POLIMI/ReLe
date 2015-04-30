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

#include "Core.h"
#include "parametric/differentiable/LinearPolicy.h"
#include "DifferentiableNormals.h"
#include "basis/IdentityBasis.h"
#include "basis/PolynomialFunction.h"
#include "features/DenseFeatures.h"

#include "LQR.h"
#include "LQRsolver.h"
#include "solvers/IrlLQRSolver.h"
#include "PolicyEvalAgent.h"
#include "algorithms/MWAL.h"
#include "policy_search/PGPE/PGPE.h"
#include "ParametricRewardMDP.h"

using namespace std;
using namespace ReLe;
using namespace arma;


int main(int argc, char *argv[])
{
    /* Learn lqr correct policy */
    mat A, B, q, r;
    A = 1;
    B = 1;
    q = 0.6;
    r = 0.4;
    vector<mat> Q, R;
    Q.push_back(q);
    R.push_back(r);
    LQR mdp(A,B,Q,R);

    IdentityBasis* pf = new IdentityBasis(0);
    DenseFeatures phi(pf);

    //Find optimal policy
    LQRsolver optimalSolver(mdp, phi);
    optimalSolver.solve();
    Policy<DenseAction, DenseState>& expertPolicy = optimalSolver.getPolicy();
    PolicyEvalAgent<DenseAction, DenseState> expert(expertPolicy);
    cout << "Optimal Policy: " << endl << expertPolicy.printPolicy() << endl;

    /* Generate LQR expert dataset */
    Core<DenseAction, DenseState> expertCore(mdp, expert);
    CollectorStrategy<DenseAction, DenseState> collection;
    expertCore.getSettings().loggerStrategy = &collection;
    expertCore.getSettings().episodeLenght = 50;
    expertCore.getSettings().testEpisodeN = 1000;
    expertCore.runTestEpisodes();


    /* Learn weight with MWAL */

    //Create features vector
    BasisFunctions rewardBF;
    rewardBF.push_back(new InverseBasis(new PolynomialFunction(0, 2))); //-State^2
    rewardBF.push_back(new InverseBasis(new PolynomialFunction(1, 2))); //-Action^2

    DenseFeatures rewardPhi(rewardBF);

    //Compute expert feature expectations
    arma::vec muE = collection.data.computefeatureExpectation(rewardPhi, mdp.getSettings().gamma);

    IRL_LQRSolver solver(mdp, phi, rewardPhi);
    solver.setTestParams(1000, 50);

    //Run MWAL
    unsigned int T = 30;

    MWAL<DenseAction, DenseState> irlAlg(T, muE, solver);
    irlAlg.run();
    arma::vec w = irlAlg.getWeights();

    cout << "Computed weights: " << endl << irlAlg.getWeights() << endl;
    Policy<DenseAction, DenseState>* pi = irlAlg.getPolicy();
    cout <<  "Policy learned" << endl << pi->printPolicy() << endl;
    delete pi;

    return 0;
}
