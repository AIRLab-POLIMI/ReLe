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
#include "parametric/differentiable/NormalPolicy.h"
#include "features/DenseFeatures.h"
#include "DifferentiableNormals.h"
#include "basis/IdentityBasis.h"
#include "basis/GaussianRbf.h"
#include "basis/PolynomialFunction.h"

#include "LQR.h"
#include "LQRsolver.h"
#include "PolicyEvalAgent.h"
#include "algorithms/GIRL.h"
#include "policy_search/PGPE/PGPE.h"
#include "ParametricRewardMDP.h"
#include "RandomGenerator.h"
#include "FileManager.h"

#include "MLE.h"

using namespace std;
using namespace ReLe;
using namespace arma;

int main(int argc, char *argv[])
{

    int nbEpisodes = atof(argv[1]);
    vec eReward(2);
    eReward(0) = atof(argv[2]);
    eReward(1) = atof(argv[3]);


    FileManager fm("mle_Policy", "test");
    fm.createDir();
    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);



    arma::mat A(1,1), B(1,1), Q(1,1), R(1,1);
    A(0,0) = 1;
    B(0,0) = 1;
    Q(0,0) = eReward(0);
    R(0,0) = eReward(1);
    std::vector<arma::mat> Qv(1, Q);
    std::vector<arma::mat> Rv(1, R);
    LQR mdp(A,B,Qv,Rv);

    IdentityBasis* pf = new IdentityBasis(0);
    DenseFeatures phi(pf);
    NormalPolicy tmpPolicy(1, phi);

    LQRsolver solver(mdp,phi,LQRsolver::Type::CLASSIC);
    solver.setRewardWeights(eReward);
    mat K = solver.computeOptSolution();
    arma::vec p = K.diag();
    std::cout << "optimal pol: " << p.t();
    tmpPolicy.setParameters(p);

    std::cout << "Rewards: ";
    for (int i = 0; i < eReward.n_elem; ++i)
    {
        std::cout << eReward(i) << " ";
    }
    std::cout << "| Params: " << tmpPolicy.getParameters().t() << std::endl;

    PolicyEvalAgent<DenseAction, DenseState> expert(tmpPolicy);

    /* Generate LQR expert dataset */
    Core<DenseAction, DenseState> expertCore(mdp, expert);
    CollectorStrategy<DenseAction, DenseState> collection;
    expertCore.getSettings().loggerStrategy = &collection;
    expertCore.getSettings().episodeLenght = mdp.getSettings().horizon;
    expertCore.getSettings().testEpisodeN = nbEpisodes;
    expertCore.runTestEpisodes();


    //save data
    Dataset<DenseAction,DenseState>& data = collection.data;
    ofstream datafile(fm.addPath("data.log"), ios_base::out);
    datafile << std::setprecision(OS_PRECISION);
    data.writeToStream(datafile);
    datafile.close();

    MVNDiagonalPolicy policy(phi);
    MLE mle(policy, data);
    double vv[] = {0,6};
    arma::vec startVal(vv,2);
    unsigned int nbIter = 100;
    arma::vec pp = mle.solve(startVal, nbIter);

    unsigned int nval = mle.getFunEvals();
    cout << "Fun Evals: " << nval << endl;
    if (nval == nbIter)
    {
        cout << "WARNING: maximum number of function evaluations reached!" << endl;
    }

    std::cerr << pp.t();
    policy.setParameters(pp);

    int count = 0;
    arma::mat F;
    for (int ep = 0; ep < nbEpisodes; ++ep)
    {
        int nbSteps = data[ep].size();
        for (int t = 0; t < nbSteps; ++t)
        {
            Transition<DenseAction, DenseState>& tr = data[ep][t];
            arma::vec aa = policy(tr.x);
            F = arma::join_horiz(F,aa);
            ++count;
        }
    }

    F.save(fm.addPath("datafit.log"), arma::raw_ascii);


    return 0;

}

