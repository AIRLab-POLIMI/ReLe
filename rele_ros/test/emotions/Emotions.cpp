/*
 * rele,
 *
 *
 * Copyright (C) 2017 Davide Tateo
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
#include "rele/approximators/basis/FrequencyBasis.h"

#include "rele/policy/parametric/differentiable/NormalPolicy.h"

#include "rele/utils/FileManager.h"

using namespace std;
using namespace arma;
using namespace ReLe;

int main(int argc, char *argv[])
{
    FileManager fm("emotions");
    fm.createDir();
    //fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);

    double df = 0.5;
    double fE = 100.0/5.0;
    int uDim = 3;

    BasisFunctions basis = FrequencyBasis::generate(0, df, fE, df, true);
   	BasisFunctions tmp = FrequencyBasis::generate(0, 0.0, fE, df, false);
    basis.insert(tmp.begin(), tmp.end(), basis.end());



    SparseFeatures phi(basis, uDim);


    return 0;
}
