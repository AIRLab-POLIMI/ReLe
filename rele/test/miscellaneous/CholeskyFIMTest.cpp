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
#include <cassert>
#include "FileManager.h"

using namespace std;
using namespace ReLe;
using namespace arma;


int main(int argc, char *argv[])
{

    FileManager fm("chol_FIM", "test");
    fm.createDir();
    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);


    arma::vec mean0;
    mean0.load(argv[1], raw_ascii);
    arma::mat chol0;
    chol0.load(argv[2], raw_ascii);

    arma::vec mean = mean0;

    //--- distribution setup
    //----- ParametricCholeskyNormal
    mat cholMtx = chol(chol0);
    ParametricCholeskyNormal dist(mean, cholMtx);
    //----- ParametricDiagonalNormal
//    ParametricDiagonalNormal dist(mean, chol0);
    //-----
    //---

    int i, nbSamples = 100000;
    arma::mat Fs(dist.getParametersSize(), dist.getParametersSize(), arma::fill::zeros);
    for (i = 0; i < nbSamples; ++i)
    {
        vec theta = dist();
        vec grad = dist.difflog(theta);
        Fs += grad * grad.t();
//        arma::uvec q = arma::find_nonfinite(grad);
//        if (q.n_elem > 0)
//            cout << "ERROR" <<endl;
    }
    Fs /= nbSamples;

    sp_mat tmp = dist.FIM();
    mat Fe(tmp);

    tmp = dist.inverseFIM();
    mat Feinv(tmp);

    Fs.save(fm.addPath("Fs.dat"), arma::raw_ascii);
    Fe.save(fm.addPath("Fe.dat"), arma::raw_ascii);
    Feinv.save(fm.addPath("Feinv.dat"), arma::raw_ascii);


    mat invt = arma::inv(Fe);
    cout << "\naccu(abs(Feinv - inv(Fe))): " << arma::accu(abs(Feinv-invt)) << endl << endl;
    assert(arma::accu(abs(Feinv-invt)) < 1e-5);


    cout << Fs.diag() << endl;
    cout << Fe.diag() << endl;


    return 0;
}
