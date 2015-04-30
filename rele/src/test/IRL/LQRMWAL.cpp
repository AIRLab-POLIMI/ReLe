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
#include "basis/QuadraticBasis.h"
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
    mat A, B, q1, r1, q2, r2;
    A = 1;
    B = 1;
    q1 = 100;
    r1 = 1;
    q2 = 1;
    r2 = 100;
    vector<mat> Q, R;
    Q.push_back(q1);
    R.push_back(r1);
    Q.push_back(q2);
    R.push_back(r2);
    LQR mdp(A,B,Q,R);
    vec wExpert(2);
    wExpert[0] = 0.6;
    wExpert[1] = 0.4;

    IdentityBasis* pf = new IdentityBasis(0);
    DenseFeatures phi(pf);

    //Find optimal policy
    LQRsolver optimalSolver(mdp, phi);
    optimalSolver.setRewardWeights(wExpert);
    optimalSolver.solve();
    Policy<DenseAction, DenseState>& expertPolicy = optimalSolver.getPolicy();
    PolicyEvalAgent<DenseAction, DenseState> expert(expertPolicy);
    cout << "Optimal Policy: " << endl << expertPolicy.printPolicy() << endl;

    /* Generate LQR expert dataset */
    optimalSolver.setTestParams(50, 1000);
    Dataset<DenseAction, DenseState>&& dataset = optimalSolver.test();


    /* Learn weight with MWAL */

    //Create features vector
    BasisFunctions rewardBF;
    vector<mat> qb1 = {q1, r1};
    vector<mat> qb2 = {q2, r2};
    vector<span> spanV = {span(0), span(1)};
    rewardBF.push_back(new InverseBasis(new QuadraticBasis(qb1, spanV))); //objective 1
    rewardBF.push_back(new InverseBasis(new QuadraticBasis(qb2, spanV))); // Objective 2

    DenseFeatures rewardPhi(rewardBF);

    //Compute expert feature expectations
    arma::vec muE = dataset.computefeatureExpectation(rewardPhi, mdp.getSettings().gamma);

    IRL_LQRSolver solver(mdp, rewardPhi, phi);
    solver.setTestParams(1000, 50);

    //Run MWAL
    unsigned int T = 1000;

    MWAL<DenseAction, DenseState> irlAlg(T, muE, solver);
    irlAlg.run();
    arma::vec w = irlAlg.getWeights();

    cout << "Computed weights: " << endl << irlAlg.getWeights() << endl;
    Policy<DenseAction, DenseState>* pi = irlAlg.getPolicy();
    cout <<  "Policy learned" << endl << pi->printPolicy() << endl;
    delete pi;

    return 0;
}
