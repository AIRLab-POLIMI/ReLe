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

#include "NLS.h"
#include "policy_search/PGPE/PGPE.h"
#include "DifferentiableNormals.h"
#include "Core.h"
#include "parametric/differentiable/NormalPolicy.h"
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
    NLS mdp;
    //with these settings
    //max in ( -3.58, 10.5 ) -> J = 8.32093
    //note that there are multiple optimal solutions
    //TODO: verificare, serve interfaccia core per valutare una politica

    int dim = mdp.getSettings().continuosStateDim;
    cout << "dim: " << dim << endl;


    //--- define meta distribution (high-level policy)
    arma::vec mean(dim, arma::fill::zeros);
    mean[0] = -0.42;
    mean[1] =  0.42;
    arma::mat cov(dim, dim, arma::fill::eye);
    cov *= 0.1;
    ParametricNormal dist(mean,cov);
    //---


    //--- define policy (low level)
    DenseBasisVector basis;
    basis.generatePolynomialBasisFunctions(1,dim);
    delete basis.at(0);
    basis.erase(basis.begin());
    cout << "--- Mean regressor ---" << endl;
    cout << basis << endl;
    LinearApproximator meanRegressor(dim, basis);

    DenseBasisVector stdBasis;
    stdBasis.generatePolynomialBasisFunctions(1,dim);
    delete stdBasis.at(0);
    stdBasis.erase(stdBasis.begin());
    cout << "--- Standard deviation regressor ---" << endl;
    cout << stdBasis << endl;
    LinearApproximator stdRegressor(dim, stdBasis);
    arma::vec stdWeights(stdRegressor.getParametersSize());
    stdWeights.fill(0.5);
    stdRegressor.setParameters(stdWeights);


    NormalStateDependantStddevPolicy policy(&meanRegressor, &stdRegressor);
    //---

    int nbepperpol = 1, nbpolperupd = 5;
    bool usebaseline = true;
    PGPE<DenseAction, DenseState> agent(dist, policy, nbepperpol, nbpolperupd, 0.1, usebaseline);
    agent.setNormalization(true);

    ReLe::Core<DenseAction, DenseState> core(mdp, agent);
    PrintStrategy<DenseAction, DenseState> stat(false);
    core.getSettings().loggerStrategy = &stat;

    int nbUpdates = 1;
    int episodes  = nbUpdates*nbepperpol*nbpolperupd;
    for (int i = 0; i < episodes; i++)
    {
        core.getSettings().episodeLenght = mdp.getSettings().horizon;
        //        cout << "starting episode" << endl;
        core.runEpisode();
    }
    agent.printStatistics("PGPEStats.txt");


//    EvaluateStrategy<DenseAction, DenseState> stat_e;
//    core.getSettings().loggerStrategy = &stat_e;
//    int testEpisodes = 10;
//    arma::vec Jm(mdp.getSettings().rewardDim, arma::fill::zeros);
//    for (int i = 0; i < testEpisodes; i++)
//    {
//        core.getSettings().episodeLenght = mdp.getSettings().horizon;
//        //        cout << "starting episode" << endl;
//        core.runTestEpisode();
////        cout << "[" << i << "]" << stat_e->J_mean.t();
//        Jm += stat_e.J_mean;
//    }
//    Jm /= testEpisodes;
//    cout << Jm;

//    cout << "Batch test\n";
//    cout << core.runBatchTest(testEpisodes);

    return 0;
}
