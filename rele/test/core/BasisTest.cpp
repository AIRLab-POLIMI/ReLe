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

#ifndef SRC_TEST_CORE_ANDCONDITIONFEATURESTEST_CPP_
#define SRC_TEST_CORE_ANDCONDITIONFEATURESTEST_CPP_

#define ARMA_USE_CXX11 //workaround for VS2015
#include <armadillo>

#include "rele/approximators/Features.h"
#include "rele/approximators/basis/IdentityBasis.h"
#include "rele/approximators/basis/ConditionBasedFunction.h"
#include "rele/approximators/basis/GaussianRbf.h"
#include "rele/approximators/features/DenseFeatures.h"

using namespace std;
using namespace ReLe;
using namespace arma;

int main(int argc, char *argv[])
{
	/* test gaussian RBF */
	cout << "Gaussias RBF Test" << endl;
	BasisFunctions basis = GaussianRbf::generate(
	{
		3,
		3
	},
	{
	    0.0, 1.0,
	    0.0, 1.0
	});

	for(auto bf : basis)
		std::cout << *bf << std::endl;

	DenseFeatures phi(basis);

	vec input1 = {0.1, 0.1};
	vec input2 = {0.5, 0.5};
	vec input3 = {0.1, 0.9};
	vec input4 = {0.9, 0.1};

	cout << "phi(0)" << endl << phi(input1) << endl;
	cout << "phi(1)" << endl << phi(input2) << endl;
	cout << "phi(2)" << endl << phi(input3) << endl;
	cout << "phi(3)" << endl << phi(input4) << endl;

    /* Test vectorFiniteIdentity */
    cout << "VectorFiniteIdentity Basis Test" << endl;
    BasisFunctions basis0 = VectorFiniteIdentityBasis::generate(1, 3);

    vec i0 = {0};
    vec i1 = {1};
    vec i2 = {2};
    vec i3 = {3};

    DenseFeatures phi0(basis0);

    cout << "phi0(0)" << endl << phi0(i0) << endl;
    cout << "phi0(1)" << endl << phi0(i1) << endl;
    cout << "phi0(2)" << endl << phi0(i2) << endl;
    cout << "phi0(3)" << endl << phi0(i3) << endl;

    /* And Condition test */
    cout << "AndCondition Basis Test 1" << endl;
    vec in0 = {0, 0, 0};
    vec in1 = {0, 0, 1};
    vec in2 = {0, 1, 0};
    vec in3 = {0, 1, 1};
    vec in4 = {1, 0, 0};
    vec in5 = {1, 0, 1};
    vec in6 = {1, 1, 0};
    vec in7 = {1, 1, 1};


    vector<unsigned int> indexes = {1, 2};
    vector<unsigned int> values = {2, 2};

    BasisFunctions basis1 = VectorFiniteIdentityBasis::generate(1, 2);
    BasisFunctions basis2 = AndConditionBasisFunction::generate(basis1, indexes, values);

    DenseFeatures phi1(basis2);

    cout << "phi1(0)" << endl << phi1(in0).t() << endl;
    cout << "phi1(1)" << endl << phi1(in1).t() << endl;
    cout << "phi1(2)" << endl << phi1(in2).t() << endl;
    cout << "phi1(3)" << endl << phi1(in3).t() << endl;
    cout << "phi1(4)" << endl << phi1(in4).t() << endl;
    cout << "phi1(5)" << endl << phi1(in5).t() << endl;
    cout << "phi1(6)" << endl << phi1(in6).t() << endl;
    cout << "phi1(7)" << endl << phi1(in7).t() << endl;

    cout << "AndCondition Basis Test 2" << endl;
    vec inp0 = {0, 0};
    vec inp1 = {0, 1};
    vec inp2 = {0, 2};
    vec inp3 = {0, 3};
    vec inp4 = {1, 0};
    vec inp5 = {1, 1};
    vec inp6 = {1, 2};
    vec inp7 = {1, 3};

    BasisFunctions basis3 = VectorFiniteIdentityBasis::generate(1, 2);
    BasisFunctions basis4 = AndConditionBasisFunction::generate(basis3, 1, 4);

    DenseFeatures phi2(basis4);

    cout << "phi2(0)" << endl << phi2(inp0).t() << endl;
    cout << "phi2(1)" << endl << phi2(inp1).t() << endl;
    cout << "phi2(2)" << endl << phi2(inp2).t() << endl;
    cout << "phi2(3)" << endl << phi2(inp3).t() << endl;
    cout << "phi2(4)" << endl << phi2(inp4).t() << endl;
    cout << "phi2(5)" << endl << phi2(inp5).t() << endl;
    cout << "phi2(6)" << endl << phi2(inp6).t() << endl;
    cout << "phi2(7)" << endl << phi2(inp7).t() << endl;

}


#endif /* SRC_TEST_CORE_ANDCONDITIONFEATURESTEST_CPP_ */
