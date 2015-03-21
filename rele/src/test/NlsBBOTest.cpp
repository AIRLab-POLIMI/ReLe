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
#include "policy_search/NES/NES.h"
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
    //max in ( many optimal points ) -> J = 8.5
    //note that there are multiple optimal solutions
    //e.g.
    //-3.2000    8.8000    8.4893
    //-3.2000    9.3000    8.4959
    //-3.2000    9.5000    8.4961
    //-3.4000   10.0000    8.5007
    //-3.2000    9.4000    8.5020
    //-3.1000    8.8000    8.5028
    //-3.4000    9.7000    8.5041
    //-3.0000    8.1000    8.5205
    //-2.9000    7.7000    8.5230
    //-3.1000    9.1000    8.5243
    //-2.8000    7.3000    8.5247

    int dim = mdp.getSettings().continuosStateDim;
    cout << "dim: " << dim << endl;


    //--- define meta distribution (high-level policy)
    arma::vec mean(dim, arma::fill::zeros);
    mean[0] = -0.42;
    mean[1] =  0.42;
//    arma::mat cov(dim, dim, arma::fill::eye);
//    cov *= 0.1;
//    ParametricNormal dist(mean,cov);
    vec sigmas(dim, fill::ones);
//    ParametricDiagonalNormal dist(mean, sigmas);
    mat cholMtx = chol(eye(dim,dim));
    ParametricCholeskyNormal dist(mean, cholMtx);
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

    int nbepperpol = 1, nbpolperupd = 40;
    bool usebaseline = true;
//    PGPE<DenseAction, DenseState> agent(dist, policy, nbepperpol, nbpolperupd, 0.05, usebaseline);
//    agent.setNormalization(true);
//    NES<DenseAction, DenseState> agent(dist, policy, nbepperpol, nbpolperupd, 0.01, usebaseline);


    double stepnb = (3.0/5.0)*(3+log(dim))/(dim*sqrt(dim));
    xNES<DenseAction, DenseState> agent(dist, policy, nbepperpol, nbpolperupd, 1.0, stepnb, stepnb);

    const std::string outfile = "nls_bbo_out.txt";
    ReLe::Core<DenseAction, DenseState> core(mdp, agent);
    core.getSettings().loggerStrategy = new WriteStrategy<DenseAction, DenseState>(
        outfile, WriteStrategy<DenseAction, DenseState>::AGENT,
        true /*delete file*/
    );

    int horiz = mdp.getSettings().horizon;
    core.getSettings().episodeLenght = horiz;

    int nbUpdates = 100;
    int episodes  = nbUpdates*nbepperpol*nbpolperupd;
    double every, bevery;
    every = bevery = 0.01; //%
    int updateCount = 0;
    for (int i = 0; i < episodes; i++)
    {
        core.runEpisode();

        int v = nbepperpol*nbpolperupd;
        if (i % v == 0)
        {
            updateCount++;
            if ((updateCount >= nbUpdates*every) || (updateCount == 1))
            {
                int p = 100 * updateCount/static_cast<double>(nbUpdates);
                cout << "### " << p << "% ###" << endl;
                cout << dist.getParameters().t();
                arma::vec J = core.runBatchTest(100);
                cout << "mean score: " << J(0) << endl;
                every += bevery;
            }
        }
    }

    int nbTestEpisodes = 1000;
    cout << "Final test [#episodes: " << nbTestEpisodes << " ]" << endl;
    cout << core.runBatchTest(1000) << endl;

    //--- collect some trajectories
    const std::string testOutfile = "nls_bbo_final_trajectories.csv";
    core.getSettings().loggerStrategy = new WriteStrategy<DenseAction, DenseState>(
        testOutfile, WriteStrategy<DenseAction, DenseState>::TRANS,
        true /*delete file*/
    );
    for (int n = 0; n < 100; ++n)
        core.runTestEpisode();
    //---

    return 0;
}
