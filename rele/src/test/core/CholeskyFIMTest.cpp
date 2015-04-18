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

#include "DifferentiableNormals.h"
#include "Core.h"
#include "RandomGenerator.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <random>
#include <cmath>

using namespace std;
using namespace ReLe;
using namespace arma;


int main(int argc, char *argv[])
{


    //--- distribution setup
    //----- ParametricCholeskyNormal
    int nparams = 3;
    arma::vec mean(nparams, fill::zeros);
    arma::mat cov(nparams, nparams, arma::fill::eye);
    mat cholMtx = chol(cov);
    ParametricCholeskyNormal dist(mean, cholMtx);
    //----- ParametricDiagonalNormal
    //    vec mean(nparams, fill::zeros);
    //    vec sigmas(nparams, fill::ones);
    //    ParametricDiagonalNormal dist(mean, sigmas);
    //-----
    //---

    int i, nbSamples = 10e6;
    arma::mat Fs(dist.getParametersSize(), dist.getParametersSize());
    for (i = 0; i < nbSamples; ++i)
    {
        vec theta = dist();
        vec grad = dist.difflog(theta);
        Fs += grad * grad.t();
    }
    Fs /= nbSamples;
    cout << Fs << endl;

    sp_mat tmp = dist.FIM();
    mat Fe(tmp);

    Fs.save("Fs.dat");
    Fe.save("Fe.dat");

    cout << Fs.diag() << endl;
    cout << Fe.diag() << endl;


    return 0;
}
