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

#include "rele/approximators/features/SparseFeatures.h"
#include "rele/approximators/features/DenseFeatures.h"
#include "rele/approximators/regressors/others/LinearApproximator.h"
#include "rele/approximators/basis/IdentityBasis.h"
#include "rele/approximators/basis/PolynomialFunction.h"

#include "rele/policy/parametric/differentiable/NormalPolicy.h"

#include "rele/environments/LQR.h"
#include "rele/solvers/lqr/LQRsolver.h"
#include "rele/core/PolicyEvalAgent.h"

#include "rele/utils/FileManager.h"

#include "../RewardBasisLQR.h"

#include <chrono>

#include "rele/IRL/utils/StepBasedHessianCalculatorFactory.h"
#include "rele/IRL/utils/StepBasedGradientCalculatorFactory.h"

using namespace std;
using namespace arma;
using namespace ReLe;

int main(int argc, char *argv[])
{
//  RandomGenerator::seed(45423424);
//  RandomGenerator::seed(8763575);

    if(argc != 5)
    {
        std::cout << "Error, you must give the policy, the baseline, the number of episodes, and index" << std::endl;
        return -1;
    }

    std::string policyName(argv[1]);
    std::string baseline(argv[2]);
    std::string episodes(argv[3]);
    std::string testN(argv[4]);

    FileManager fm("lqrPrint/" + baseline + "/" + episodes  + "/" + testN);
    fm.createDir();

    unsigned int nbEpisodes = std::stoul(episodes);

    IrlGrad atype = IrlGrad::REINFORCE_BASELINE;
    IrlHess htype;
    bool isStateDependant = false;

    if(policyName == "normal")
        isStateDependant = false;
    else if(policyName == "stateDep")
        isStateDependant = true;


    if(baseline == "normal")
        htype = IrlHess::REINFORCE_BASELINE;
    else if(baseline == "trace")
        htype = IrlHess::REINFORCE_BASELINE_TRACE;
    else if(baseline == "diag")
        htype = IrlHess::REINFORCE_BASELINE_TRACE_DIAG;
    else
    {
        std::cout << "Wrong baseline specified" << std::endl;
        return -1;
    }

    vec eReward =
    { 0.3, 0.7 };
    int dim = eReward.n_elem;



    // create policy basis functions
    BasisFunctions basis = IdentityBasis::generate(dim);
    SparseFeatures phi;
    phi.setDiagonal(basis);

    BasisFunctions basisStdDev = PolynomialFunction::generate(1, dim);
    SparseFeatures phiStdDev(basisStdDev, dim);
    arma::mat stdDevW(dim, phiStdDev.rows(), fill::zeros);

    for(int i = 0; i < stdDevW.n_rows; i++)
    {
        int first = i*basisStdDev.size();
        stdDevW(i, first) = 0.1;
        for(int j =  first + 1; j < (i+1)*basisStdDev.size(); j++)
        {
            stdDevW(i, j) = 0.1;
        }
    }

    std::cout << basisStdDev.size() << std::endl;
    std::cout << phiStdDev.rows() << ", " << phiStdDev.cols() << std::endl;
    std::cout << stdDevW << std::endl;


    // solve the problem in exact way
    LQR mdp(dim, dim);
    LQRsolver solver(mdp, phi);
    solver.setRewardWeights(eReward);
    mat K = solver.computeOptSolution();
    arma::vec p = K.diag();

    // Create expert policy
    DifferentiablePolicy<DenseAction, DenseState>* pol;

    if(isStateDependant)
    {
        pol = new MVNStateDependantStddevPolicy(phi, phiStdDev, stdDevW);
    }
    else
    {
        arma::mat SigmaExpert(dim, dim, arma::fill::eye);
        SigmaExpert *= 0.1;
        pol = new MVNPolicy(phi, SigmaExpert);
    }

    DifferentiablePolicy<DenseAction, DenseState>& expertPolicy = *pol;
    expertPolicy.setParameters(p);

    std::cout << "Rewards: ";
    for (int i = 0; i < eReward.n_elem; ++i)
    {
        std::cout << eReward(i) << " ";
    }
    std::cout << "| Params: " << expertPolicy.getParameters().t() << std::endl;

    PolicyEvalAgent<DenseAction, DenseState> expert(expertPolicy);

    // Generate LQR expert dataset
    Core<DenseAction, DenseState> expertCore(mdp, expert);
    CollectorStrategy<DenseAction, DenseState> collection;
    expertCore.getSettings().loggerStrategy = &collection;
    expertCore.getSettings().episodeLength = mdp.getSettings().horizon;
    expertCore.getSettings().testEpisodeN = nbEpisodes;
    expertCore.runTestEpisodes();
    Dataset<DenseAction, DenseState>& data = collection.data;

    // Create parametric reward
    BasisFunctions basisReward;
    for (unsigned int i = 0; i < eReward.n_elem; i++)
        basisReward.push_back(new LQR_RewardBasis(i, dim));
    DenseFeatures phiReward(basisReward);


    // init calculators methods
    auto gradientCalculator = StepBasedGradientCalculatorFactory<DenseAction, DenseState>::build(atype,
                              phiReward, data, expertPolicy,
                              mdp.getSettings().gamma);

    auto hessianCalculator = StepBasedHessianCalculatorFactory<DenseAction, DenseState>::build(htype,
                             phiReward, data, expertPolicy,
                             mdp.getSettings().gamma);

    arma::vec rvec = data.computefeatureExpectation(phiReward, mdp.getSettings().gamma);


    unsigned int samplesParams = 101;
    arma::vec valuesJ(samplesParams, arma::fill::zeros);
    arma::vec valuesG(samplesParams, arma::fill::zeros);
    arma::vec valuesT(samplesParams, arma::fill::zeros);
    arma::vec valuesF(samplesParams, arma::fill::zeros);
    arma::vec valuesFs(samplesParams, arma::fill::zeros);
    arma::mat valuesE(samplesParams, 2, arma::fill::zeros);


    // sample functions
    for (int i = 0; i < samplesParams; i++)
    {
        double w1 = i / 100.0;
        arma::vec w = { w1, 1.0 - w1 };

        // compute gradient and hessian
        arma::vec g = gradientCalculator->computeGradient(w);
        arma::mat H = hessianCalculator->computeHessian(w);

        // compute signed hessian
        arma::mat V;
        arma::vec Lambda;
        arma::eig_sym(Lambda, V, H);

        arma::mat Hs = V*arma::diagmat(arma::abs(Lambda))*V.i();

        //compute the sigma matrix
        double eps = 0;
        arma::mat Sigma = arma::eye(2, 2)*eps;


        // compute J
        valuesJ.row(i) = rvec.t()*w;

        // compute gradient norm
        valuesG.row(i) = g.t()*g;

        //compute trace of the hessian
        valuesT.row(i) = arma::trace(H);

        //compute expectedDeltaIRL function
        valuesF.row(i) = -0.5*g.t()*H.i()*g+0.5*arma::trace(H*Sigma);

        //compute the signed expectedDeltaIRL function
        valuesFs.row(i) = g.t()*Hs.i()*g + 0.5*g.t()*H.i()*g + 0.5*arma::trace(H*Sigma);

        //save eigenvalues
        valuesE(i, 0) = Lambda(0);
        valuesE(i, 1) = Lambda(1);

    }

    std::cout << "Saving results" << std::endl;

    valuesJ.save(fm.addPath("J.txt"), arma::raw_ascii);
    valuesG.save(fm.addPath("G.txt"), arma::raw_ascii);
    valuesT.save(fm.addPath("T.txt"), arma::raw_ascii);
    valuesF.save(fm.addPath("F.txt"), arma::raw_ascii);
    valuesFs.save(fm.addPath("Fs.txt"), arma::raw_ascii);
    valuesE.save(fm.addPath("E.txt"), arma::raw_ascii);

    std::cout << "Work complete" << std::endl;

    delete pol;


    return 0;
}
