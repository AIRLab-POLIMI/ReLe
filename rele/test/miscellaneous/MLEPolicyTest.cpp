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

#include "rele/core/Core.h"
#include "rele/policy/parametric/differentiable/NormalPolicy.h"
#include "rele/approximators/features/SparseFeatures.h"
#include "rele/approximators/features/DenseFeatures.h"
#include "rele/approximators/basis/IdentityBasis.h"
#include "rele/approximators/basis/GaussianRbf.h"
#include "rele/approximators/basis/PolynomialFunction.h"

#include "rele/environments/LQR.h"
#include "rele/solvers/lqr/LQRsolver.h"
#include "rele/core/PolicyEvalAgent.h"
#include "rele/algorithms/policy_search/PGPE/PGPE.h"
#include "rele/IRL/ParametricRewardMDP.h"
#include "rele/utils/RandomGenerator.h"
#include "rele/utils/FileManager.h"

#include "rele/policy/utils/MLE.h"

using namespace std;
using namespace ReLe;
using namespace arma;

/*
 * 1. policy type
 * 2. start point file
 * 3. nbEpisodes
 * 4. weights
 * 5. policy params
 */
int main(int argc, char *argv[])
{

    int nbEpisodes = atof(argv[3]);
    vec eReward;
    eReward.load(argv[4], arma::raw_ascii);


    FileManager fm("mle_Policy", "test");
    fm.createDir();
    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);

    int dim = 1;
    assert(eReward.n_elem == 2);
    arma::mat A(1,1), B(1,1), Q(1,1), R(1,1);
    A(0,0) = 1;
    B(0,0) = 1;
    Q(0,0) = eReward(0);
    R(0,0) = eReward(1);
    std::vector<arma::mat> Qv(1, Q);
    std::vector<arma::mat> Rv(1, R);
    LQR mdp(A,B,Qv,Rv);


    BasisFunctions basisOrig;
    for (int i = 0; i < dim; ++i)
    {
        basisOrig.push_back(new IdentityBasis(i));
    }
    SparseFeatures phiOrig;
    phiOrig.setDiagonal(basisOrig);

    BasisFunctions basispol = PolynomialFunction::generate(2,dim);
    SparseFeatures phipol(basispol,dim);

    BasisFunctions basisrbf = GaussianRbf::generate({5}, {-4,4});
    SparseFeatures phirbf(basisrbf,dim);

    arma::mat cov(dim,dim, arma::fill::eye);
    cov *= 2;

    DifferentiablePolicy<DenseAction, DenseState>* tmpPolicy;

    if (strcmp(argv[1], "mvndiag")==0)
    {
        tmpPolicy = new MVNDiagonalPolicy(phiOrig);
    }
    else if (strcmp(argv[1], "mvnrbf")==0)
    {
        tmpPolicy = new MVNPolicy(phirbf, cov);
    }
    else if (strcmp(argv[1], "mvnpoly")==0)
    {
        tmpPolicy = new MVNPolicy(phipol, cov);
    }

    arma::vec ll;
    ll.load(argv[5], arma::raw_ascii);
    tmpPolicy->setParameters(ll);


//    IdentityBasis* pf = new IdentityBasis(0);
//    DenseFeatures phi(pf);
//    NormalPolicy tmpPolicy(1, phi);


//    LQRsolver solver(mdp,phi,LQRsolver::Type::CLASSIC);
//    solver.setRewardWeights(eReward);
//    mat K = solver.computeOptSolution();
//    arma::vec p = K.diag();
//    std::cout << "optimal pol: " << p.t();
//    tmpPolicy.setParameters(p);

    std::cout << "Rewards: ";
    for (int i = 0; i < eReward.n_elem; ++i)
    {
        std::cout << eReward(i) << " ";
    }
    std::cout << "| Params: " << tmpPolicy->getParameters().t() << std::endl;

    PolicyEvalAgent<DenseAction, DenseState> expert(*tmpPolicy);

    /* Generate LQR expert dataset */
    Core<DenseAction, DenseState> expertCore(mdp, expert);
    CollectorStrategy<DenseAction, DenseState> collection;
    expertCore.getSettings().loggerStrategy = &collection;
    expertCore.getSettings().episodeLength = mdp.getSettings().horizon;
    expertCore.getSettings().testEpisodeN = nbEpisodes;
    expertCore.runTestEpisodes();


    //save data
    Dataset<DenseAction,DenseState>& data = collection.data;
    ofstream datafile(fm.addPath("data.log"), ios_base::out);
    datafile << std::setprecision(OS_PRECISION);
    data.writeToStream(datafile);
    datafile.close();

    DifferentiablePolicy<DenseAction, DenseState>* policy;

    if (strcmp(argv[1], "mvndiag")==0)
    {
        policy = new MVNDiagonalPolicy(phiOrig);
    }
    else if (strcmp(argv[1], "mvnrbf")==0)
    {
        policy = new MVNPolicy(phirbf, cov);
    }
    else if (strcmp(argv[1], "mvnpoly")==0)
    {
        policy = new MVNPolicy(phipol, cov);
    }



    MLE<DenseAction,DenseState> mle(*policy, data);
    arma::vec startVal;
    startVal.load(argv[2], arma::raw_ascii);
    unsigned int nbIter = 100;
    double logLikelihood = mle.compute(startVal, nbIter);
    arma::vec pp = policy->getParameters();

    pp.save(fm.addPath("mleparams.log"), arma::raw_ascii);

    std::cerr << pp.t();
    policy->setParameters(pp);

    int count = 0;
    arma::mat F;
    for (int ep = 0; ep < nbEpisodes; ++ep)
    {
        int nbSteps = data[ep].size();
        for (int t = 0; t < nbSteps; ++t)
        {
            Transition<DenseAction, DenseState>& tr = data[ep][t];
            arma::vec aa = (*policy)(tr.x);
            F = arma::join_horiz(F,aa);
            ++count;
        }
    }
    F.save(fm.addPath("datafit.log"), arma::raw_ascii);

    delete policy;

    return 0;

}

