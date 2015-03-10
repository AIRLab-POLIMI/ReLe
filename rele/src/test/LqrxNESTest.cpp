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

#include "LQR.h"
#include "policy_search/NES/NES.h"
#include "DifferentiableNormals.h"
#include "Core.h"
#include "parametric/differentiable/LinearPolicy.h"
#include "BasisFunctions.h"
#include "basis/PolynomialFunction.h"
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
//    int dff = 4;
//    arma::vec meand(dff,arma::fill::zeros);
//    arma::mat covd = 12 * arma::eye(dff,dff);
//    arma::mat cholm = arma::chol(covd);
//    ParametricCholeskyNormal dd(dff, meand,cholm);
//    arma::sp_mat fim = dd.FIM();
//    cout << fim << endl;
//    arma::sp_mat invfim = dd.inverseFIM();
//    cout << invfim << endl;

    LQR mdp(1,1); //with these settings the optimal value is -0.6180 (for the linear policy)

    arma::vec mean(1);
    mean[0] = -0.1;
    arma::mat cov(1,1, arma::fill::eye);
    cov *= 0.01;
    ParametricCholeskyNormal dist(1, mean, cov);


    PolynomialFunction* pf = new PolynomialFunction(1,1);
    cout << *pf << endl;
    DenseBasisVector basis;
    basis.push_back(pf);
    cout << basis << endl;
    LinearApproximator regressor(mdp.getSettings().continuosStateDim, basis);

    arma::vec init_params(1);
    init_params[0] = -0.1;

    regressor.setParameters(init_params);
    DetLinearPolicy<DenseState> policy(&regressor);

    int nbepperpol = 1, nbpolperupd = 10;
    bool usebaseline = true;
    xNES<DenseAction, DenseState, ParametricCholeskyNormal> agent(dist, policy, nbepperpol, nbpolperupd, 0.001, usebaseline);
    agent.setNormalization(true);

    ReLe::Core<DenseAction, DenseState> core(mdp, agent);

    int nbUpdates = 2000;
    int episodes  = nbUpdates*nbepperpol*nbpolperupd;
    for (int i = 0; i < episodes; i++)
    {
        core.getSettings().episodeLenght = 50;
        //        cout << "starting episode" << endl;
        core.runEpisode();
    }
    agent.printStatistics("PGPEStats.txt");

    return 0;
}
