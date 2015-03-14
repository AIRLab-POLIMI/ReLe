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
#include "policy_search/PGPE/PGPE.h"
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
    LQR mdp(1,1); //with these settings the optimal value is -0.6180 (for the linear policy)

//    arma::vec mean(1);
//    mean[0] = -0.1;
//    arma::mat cov(1,1, arma::fill::eye);
//    cov *= 0.01;
//    ParametricNormal dist(mean,cov);

    arma::vec mean(1);
    mean[0] = -0.1;
    arma::mat cov(1,1, arma::fill::eye);
    cov *= 0.01;
    mat cholMtx = chol(cov);
    ParametricCholeskyNormal dist(1, mean, cholMtx);

    PolynomialFunction* pf = new PolynomialFunction(1,1);
    cout << *pf << endl;
    DenseBasisVector basis;
    basis.push_back(pf);
    cout << basis << endl;
    LinearApproximator regressor(mdp.getSettings().continuosStateDim, basis);
    DetLinearPolicy<DenseState> policy(&regressor);

    int nbepperpol = 1, nbpolperupd = 50;
    bool usebaseline = false;
    PGPE<DenseAction, DenseState> agent(dist, policy, nbepperpol, nbpolperupd, 0.002, usebaseline);
    agent.setNormalization(true);

    ReLe::Core<DenseAction, DenseState> core(mdp, agent);
//    PrintStrategy<DenseAction, DenseState> stat(false);
//    core.getSettings().loggerStrategy = &stat;
    core.getSettings().loggerStrategy = new WriteStrategy<DenseAction, DenseState>("prova.txt", WriteStrategy<DenseAction, DenseState>::AGENT);


    std::ofstream ofs("prova.txt", std::ios_base::out);
    ofs.close();

    int nbUpdates = 4000;
    int episodes  = nbUpdates*nbepperpol*nbpolperupd;

    double every, bevery = 0.05; //10%
    every = bevery;
    int updateCount = 0;

    for (int i = 0; i < episodes; i++)
    {
        core.getSettings().episodeLenght = 50;
        //        cout << "starting episode" << endl;
        core.runEpisode();

        int v = nbepperpol*nbpolperupd;
        if (i % v == 0)
        {
            updateCount++;
            if (updateCount >= nbUpdates*every)
            {
                int p = 100 * updateCount/static_cast<double>(nbUpdates);
                cout << "### " << p << "% ###" << endl;
                cout << dist.getParameters().t();
                every += bevery;
            }
        }
    }

    cout << "Final meta distribution: " << dist.getParameters().t();

    return 0;
}
