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

#include "nonparametric/RandomPolicy.h"
#include "RandomGenerator.h"
#include "FileManager.h"
#include "basis/IdentityBasis.h"
#include "basis/PolynomialFunction.h"
#include "basis/GaussianRbf.h"
#include "features/DenseFeatures.h"
#include "features/SparseFeatures.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <cmath>

using namespace std;
using namespace ReLe;
using namespace arma;

int main(int argc, char *argv[])
{
    FileManager fm("rand", "testpol");
    fm.createDir();
    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);

    vector<FiniteAction> actions;
    for (int i = 0; i < 3; ++i)
        actions.push_back(FiniteAction(i));
    StochasticDiscretePolicy<FiniteAction, DenseState> policy(actions);

    vec state(1);
    int dim = 5000;
    vec samples(dim);
    for (int i = 0; i < dim; ++i)
    {
        state(0) = RandomGenerator::sampleUniform(-3,3);
        samples(i) = policy(state);
    }

    samples.save(fm.addPath("actions.log"), arma::raw_ascii);

    return 0;
}
