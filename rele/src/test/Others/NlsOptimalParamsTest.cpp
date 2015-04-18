/*
 * rele,
 *
 *
 * Copyright (C) 2015  Davide Tateo & Matteo Pirotta
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

#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <random>
#include <cmath>

#include "NLS.h"
#include "DifferentiableNormals.h"
#include "Core.h"
#include "parametric/differentiable/NormalPolicy.h"
#include "BasisFunctions.h"
#include "basis/PolynomialFunction.h"
#include "RandomGenerator.h"
#include "PolicyEvalAgent.h"

using namespace std;
using namespace ReLe;
using namespace arma;

int main(int argc, char *argv[])
{
    NLS mdp;

    int dim = mdp.getSettings().continuosStateDim;
    cout << "dim: " << dim << endl;

    //--- define policy (low level)
    BasisFunctions basis = PolynomialFunction::generatePolynomialBasisFunctions(1,dim);
    delete basis.at(0);
    basis.erase(basis.begin());
    DenseFeatures phi(basis);
    LinearApproximator meanRegressor(dim, phi);

    BasisFunctions stdBasis = PolynomialFunction::generatePolynomialBasisFunctions(1,dim);
    delete stdBasis.at(0);
    stdBasis.erase(stdBasis.begin());
    DenseFeatures stdPhi(stdBasis);
    LinearApproximator stdRegressor(dim, stdPhi);
    arma::vec stdWeights(stdRegressor.getParametersSize());
    stdWeights.fill(0.5);
    stdRegressor.setParameters(stdWeights);


    NormalStateDependantStddevPolicy policy(&meanRegressor, &stdRegressor);
    //---

    PolicyEvalAgent<DenseAction, DenseState> agent(policy);
    ReLe::Core<DenseAction, DenseState> core(mdp, agent);
    core.getSettings().episodeLenght = mdp.getSettings().horizon;

    ofstream out("NLS_OptParamSpace.dat", ios_base::out);

    if (out.is_open())
    {
        arma::vec w(2);
        for (double p1 = -10; p1 < 0; p1 += 0.1)
        {
            w[0] = p1;
            for (double p2 = 0; p2 < 15; p2 += 0.1)
            {
                w[1] = p2;
                policy.setParameters(w);
                int testEpisodes = 1000;
                core.getSettings().testEpisodeN = testEpisodes;
                arma::vec J = core.runBatchTest();
                out << p1 << "\t" << p2 <<  "\t" << J[0] << std::endl;
                cout << p1 << "\t" << p2 <<  "\t" << J[0] << std::endl;
            }
        }
        out.close();
    }
    return 0;
}
