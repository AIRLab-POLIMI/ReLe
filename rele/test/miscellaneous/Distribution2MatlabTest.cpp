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

#include "rele/statistics/DifferentiableNormals.h"
#include "rele/utils/RandomGenerator.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <cmath>

using namespace std;
using namespace ReLe;
using namespace arma;

void help(char* argv[])
{
    cout << "### Distribution Test ###" << endl;
}

int main(int argc, char *argv[])
{

    if ((argc == 0) || (argc < 5))
    {
        cout << argc << endl;
        help(argv);
        return 0;
    }

    //--- distribution setup
    DifferentiableDistribution* dist;

    arma::vec p1;
    p1.load(argv[2], raw_ascii);
    arma::mat p2;
    p2.load(argv[3], raw_ascii);


    arma::mat point;
    if (strcmp(argv[1], "normal") == 0)
    {
        //----- ParametricNormal
        dist = new ParametricNormal(p1, p2);
        point.load(argv[4], raw_ascii);
    }
    else if (strcmp(argv[1], "log") == 0)
    {
        //----- ParametricLogisticNormal
        vec varas;
        varas.load(argv[4], raw_ascii);
        dist = new ParametricLogisticNormal(p1,p2,varas);
        point.load(argv[5], raw_ascii);
    }
    else if (strcmp(argv[1], "chol") == 0)
    {
        //----- ParametricCholeskyNormal
        dist = new ParametricCholeskyNormal(p1, p2);
        point.load(argv[4], raw_ascii);
    }
    else if (strcmp(argv[1], "diag") == 0)
    {
        //----- ParametricDiagonalNormal
        dist = new ParametricDiagonalNormal(p1, p2);
        point.load(argv[4], raw_ascii);
    }

    int dim = point.n_elem, nbs = 50000;

    //draw random points
    mat P(dim,nbs);
    for (int i = 0; i < nbs; ++i)
    {
        vec sample = (*dist)();
        for (int j = 0; j < point.n_elem; ++j)
        {
            P(j,i) = sample(j);
        }
    }
    P.save("/tmp/dist2matlab/samples.dat", raw_ascii);

    //compute gradient
    vec grad = dist->difflog(point);
    grad.save("/tmp/dist2matlab/grad.dat", raw_ascii);

    //compute hessian
    mat hess = dist->diff2log(point);
    hess.save("/tmp/dist2matlab/hess.dat", raw_ascii);

    return 0;
}
