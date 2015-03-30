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
#include "BasisFunctions.h"
#include "basis/PolynomialFunction.h"

#include "LQR.h"
#include "PolicyEvalAgent.h"
#include "algorithms/MWAL.h"

using namespace std;
using namespace ReLe;
using namespace arma;

int main(int argc, char *argv[])
{
	/* Learn lqr correct policy */

    LQR mdp(1,1); //with these settings the optimal value is -0.6180 (for the linear policy)

    PolynomialFunction* pf = new PolynomialFunction(1,1);
    DenseBasisVector basis;
    basis.push_back(pf);
    LinearApproximator regressor(mdp.getSettings().continuosStateDim, basis);
    DetLinearPolicy<DenseState> policy(&regressor);
    vec param(1);
    param[0] = -0.6180;
    policy.setParameters(param);
    PolicyEvalAgent<DenseAction, DenseState, DetLinearPolicy<DenseState>> expert(policy);

    /* Generate LQR expert dataset */
    Core<DenseAction, DenseState> expertCore(mdp, expert);
    CollectorStrategy<DenseAction, DenseState> collection;
    expertCore.getSettings().loggerStrategy = &collection;
    expertCore.getSettings().episodeLenght = 50;
    expertCore.getSettings().testEpisodeN = 1000;
    expertCore.runTestEpisodes();


    /* Learn weight with MWAL */
    cout << "computed feature expectation" << endl;
    arma::vec muE = collection.data.computefeatureExpectation(basis, mdp.getSettings().gamma);
    cout << muE << endl;

    collection.data[0].print(cout);

    //MWAL<DenseAction, DenseState> irlAlg;



    return 0;
}
