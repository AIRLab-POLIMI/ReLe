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

#include "rele/policy/parametric/differentiable/GenericGibbsPolicy.h"
#include "rele/approximators/basis/IdentityBasis.h"
#include "rele/approximators/features/SparseFeatures.h"
#include "rele/approximators/basis/ConditionBasedFunction.h"
#include "rele/approximators/features/DenseFeatures.h"
#include "rele/approximators/regressors/others/LinearApproximator.h"

#include "rele/utils/NumericalGradient.h"

#include "rele/statistics/ArmadilloPDFs.h"

using namespace ReLe;

int main(int argc, char *argv[])
{
    BasisFunctions basis = IdentityBasis::generate(2);
    assert(basis.size()==2);
    BasisFunctions bfs;

    std::vector<FiniteAction> actions;
    actions.push_back(FiniteAction(0));
    actions.push_back(FiniteAction(1));
    actions.push_back(FiniteAction(2));

    for (int i = 0; i < actions.size(); ++i)
    {
        bfs.push_back(new AndConditionBasisFunction(basis[0], 2, i));
        bfs.push_back(new AndConditionBasisFunction(basis[1], 2, i));
    }

    DenseFeatures phi(bfs);
    assert(phi.rows() == 6);
    assert(phi.cols() == 1);

    arma::vec w(6, arma::fill::ones);
    LinearApproximator regressor(phi);
    regressor.setParameters(w);


    GenericParametricGibbsPolicyAllPref<DenseState> policy(actions, regressor, 1.0);

    arma::vec input= mvnrand({0.0, 0.0}, arma::diagmat(arma::vec({10.0, 10.0})));
    DenseState sinput(2);
    sinput(0) = input(0);
    sinput(1) = input(1);
    FiniteAction output = policy(input);
    arma::vec diff = policy.diff(sinput, output);
    arma::vec numDiff = arma::vectorise(NumericalGradient::compute(policy, policy.getParameters(), sinput, output));

    std::cout << "input       : " << input.t();
    std::cout << "output      :    " << output << std::endl;
    std::cout << "gradient    : " << diff.t();
    std::cout << "num gradient: " << numDiff.t();
    for (unsigned int i = 0; i < diff.n_elem; ++i)
        assert(abs(diff[i]-numDiff[i]) <= 0.001);



}
