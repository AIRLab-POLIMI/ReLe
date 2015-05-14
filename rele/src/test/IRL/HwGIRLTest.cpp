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
#include "PolicyEvalAgent.h"

#include "parametric/differentiable/NormalPolicy.h"
#include "features/DenseFeatures.h"
#include "features/SparseFeatures.h"
#include "basis/IdentityBasis.h"
#include "basis/GaussianRbf.h"
#include "basis/PolynomialFunction.h"
#include "basis/ConditionBasedFunction.h"

#include "ParametricRewardMDP.h"
#include "RandomGenerator.h"
#include "FileManager.h"

#include "MLE.h"
#include "algorithms/PGIRL.h"

#include <boost/timer/timer.hpp>

using namespace boost::timer;
using namespace std;
using namespace ReLe;
using namespace arma;

int main(int argc, char *argv[])
{
    //    RandomGenerator::seed(4265674);

    FileManager fm("hw", "GIRL");
    fm.createDir();
    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);

    unsigned int nbs = 3, nba = 1;

    /*** define policy for MLE ***/
//    BasisFunctions basis = PolynomialFunction::generate(3,nbs);
    BasisFunctions basis = GaussianRbf::generate({4,4,4}, {2,0,-2.14,2.14,-3.14,3.14});
    //create basis vector
    DenseFeatures phi(basis);
    //create policy
    NormalPolicy mlePolicy(0.5,phi);

    /*** load expert's trajectories ***/
    ifstream is("/tmp/ReLe/datahw.dat");
    Dataset<DenseAction,DenseState> dataExpert;
    dataExpert.readFromStream(is);
    is.close();

    /*** compute MLE ***/
    //    RidgeRegularizedMLE<FiniteAction,DenseState> mle(mlePolicy, dataExpert, 0.1);
    MLE<DenseAction,DenseState> mle(mlePolicy, dataExpert);
    arma::vec startVal(mlePolicy.getParametersSize(),arma::fill::ones);
    arma::vec pp = mle.solve(startVal);

    std::cerr << pp.t();
    mlePolicy.setParameters(pp);

#if 1
    arma::vec A;
    for (auto ep : dataExpert)
    {
        for (auto tr : ep)
        {
            vec ac = mlePolicy(tr.x);
            A = join_vert(A,ac);
        }
    }
    A.save("/tmp/ReLe/actions.dat", raw_ascii);
#endif

    return 0;
}
