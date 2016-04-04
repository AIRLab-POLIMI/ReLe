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
#include "rele/solvers/lqr/LQRExact.h"
#include "rele/core/PolicyEvalAgent.h"

#include "rele/utils/FileManager.h"

#include "../RewardBasisLQR.h"

#include "rele/IRL/utils/StepBasedHessianCalculatorFactory.h"
#include "rele/IRL/utils/StepBasedGradientCalculatorFactory.h"

using namespace std;
using namespace arma;
using namespace ReLe;

int main(int argc, char *argv[])
{
    FileManager fm("lqrPrint/exact/");
    fm.createDir();

    vec eReward =
    { 0.3, 0.7 };
    int dim = eReward.n_elem;

    // create policy basis functions
    BasisFunctions basis = IdentityBasis::generate(dim);
    SparseFeatures phi;
    phi.setDiagonal(basis);

    // solve the problem in exact way
    LQR mdp(dim, dim);
    LQRsolver solver(mdp, phi);
    solver.setRewardWeights(eReward);
    mat K = solver.computeOptSolution();
    arma::vec p = K.diag();
    arma::mat SigmaExpert(dim, dim, arma::fill::eye);
    SigmaExpert *= 0.1;

    std::cout << "optimal K: " << std::endl;
    std::cout << K << std::endl;

    std::cout << "optimal p: " << std::endl;
    std::cout << p << std::endl;

    // create exact gradient object
    LQRExact lqrExact(mdp);

    unsigned int samplesParams = 101;
    arma::vec valuesJ(samplesParams, arma::fill::zeros);
    arma::vec valuesG(samplesParams, arma::fill::zeros);
    arma::vec valuesT(samplesParams, arma::fill::zeros);
    arma::vec valuesF(samplesParams, arma::fill::zeros);
    arma::vec valuesFs(samplesParams, arma::fill::zeros);
    arma::mat valuesE(samplesParams, 2, arma::fill::zeros);

    // Test the solver
    arma::vec gOpt = lqrExact.computeJacobian(-p, SigmaExpert)*eReward;
    arma::vec rOpt = lqrExact.computeJ(-p, SigmaExpert);
    double rewOpt = arma::as_scalar(rOpt.t()*eReward);

    arma::vec p2 = -(p-0.0001*gOpt);
    arma::vec gOpt2 = lqrExact.computeJacobian(p2, SigmaExpert)*eReward;
    arma::vec rOpt2 = lqrExact.computeJ(p2, SigmaExpert);
    double rewOpt2 = arma::as_scalar(rOpt2.t()*eReward);

    std::cout << "gOpt = " << gOpt.t() << std::endl;
    std::cout << "rOpt = " << rOpt.t() << std::endl;
    std::cout << "rewOpt = " << rewOpt << std::endl;

    std::cout << "p2 = " << p2.t() << std::endl;
    std::cout << "gOpt2 = " << gOpt2.t() << std::endl;
    std::cout << "rOpt2 = " << rOpt2.t() << std::endl;
    std::cout << "rewOpt2 = " << rewOpt2 << std::endl;


    // sample functions
    for (int i = 0; i < samplesParams; i++)
    {
        double w1 = i / 100.0;
        arma::vec w = { w1, 1.0 - w1 };

        // compute gradient and hessian
        arma::mat dJ = lqrExact.computeJacobian(p, SigmaExpert);
        arma::vec g = dJ.t()*w;
        arma::mat H(dim, dim, arma::fill::zeros);
        for(unsigned int r = 0; r < dim; r++)
            H += lqrExact.computeHesian(p, SigmaExpert, r)*w(r);

        // compute signed hessian
        arma::mat V;
        arma::vec Lambda;
        arma::eig_sym(Lambda, V, H);

        arma::mat Hs = V*arma::diagmat(arma::abs(Lambda))*V.i();

        //compute the sigma matrix
        double eps = 0;
        arma::mat Sigma = arma::eye(2, 2)*eps;


        // compute J
        valuesJ.row(i) = lqrExact.computeJ(p, SigmaExpert).t()*w;

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


    return 0;
}
