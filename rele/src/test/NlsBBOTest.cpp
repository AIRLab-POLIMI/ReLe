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

#include "policy_search/PGPE/PGPE.h"
#include "policy_search/NES/NES.h"
#include "policy_search/REPS/REPS.h"
#include "DifferentiableNormals.h"
#include "Core.h"
#include "parametric/differentiable/NormalPolicy.h"
#include "BasisFunctions.h"
#include "basis/PolynomialFunction.h"
#include "RandomGenerator.h"
#include "FileManager.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <random>
#include <cmath>
#include "NLS.h"

using namespace std;
using namespace ReLe;
using namespace arma;

int main(int argc, char *argv[])
{
    FileManager fm("Nls", "BBO");
    fm.createDir();
    fm.cleanDir();

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

    //--- distribution setup
    int nparams = basis.size();
    arma::vec mean(nparams, fill::zeros);
    mean[0] = -0.42;
    mean[1] =  0.42;

    //----- ParametricNormal
    //    arma::mat cov(nparams, nparams, arma::fill::eye);
    //    ParametricNormal dist(mean, cov);
    //----- ParametricLogisticNormal
    //    ParametricLogisticNormal dist(mean, zeros(nparams), 1);
    //----- ParametricCholeskyNormal
    arma::mat cov(nparams, nparams, arma::fill::eye);
    mat cholMtx = chol(cov);
    ParametricCholeskyNormal dist(mean, cholMtx);
    //----- ParametricDiagonalNormal
    //    vec sigmas(nparams, fill::ones);
    //    ParametricDiagonalNormal dist(mean, sigmas);
    //-----
    //---
    AdaptiveStep steprule(0.1);

    int nbepperpol = 1, nbpolperupd = 40;
    bool usebaseline = true;
    //    PGPE<DenseAction, DenseState> agent(dist, policy, nbepperpol, nbpolperupd, steprule, usebaseline);
    //    agent.setNormalization(true);
    NES<DenseAction, DenseState> agent(dist, policy, nbepperpol, nbpolperupd, steprule, usebaseline);
    //    REPS<DenseAction, DenseState, ParametricNormal> agent(dist,policy,nbepperpol,nbpolperupd);
    //    agent.setEps(0.3);


    //    double stepnb = (3.0/5.0)*(3+log(dim))/(dim*sqrt(dim));
    //    xNES<DenseAction, DenseState> agent(dist, policy, nbepperpol, nbpolperupd, 1.0, stepnb, stepnb);

    ReLe::Core<DenseAction, DenseState> core(mdp, agent);
    core.getSettings().loggerStrategy = new WriteStrategy<DenseAction, DenseState>(
        fm.addPath("Nls.log"),
        WriteStrategy<DenseAction, DenseState>::AGENT,
        true /*delete file*/
    );

    int horiz = mdp.getSettings().horizon;
    core.getSettings().episodeLenght = horiz;

    int nbUpdates = 400;
    int episodes  = nbUpdates*nbepperpol*nbpolperupd;
    double every, bevery;
    every = bevery = 0.1; //%
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
                core.getSettings().testEpisodeN = 100;
                arma::vec J = core.runBatchTest();
                cout << "mean score: " << J(0) << endl;
                every += bevery;
            }
        }
    }

    int nbTestEpisodes = 1000;
    cout << "Final test [#episodes: " << nbTestEpisodes << " ]" << endl;
    core.getSettings().testEpisodeN = 1000;
    cout << core.runBatchTest() << endl;

    //--- collect some trajectories
    core.getSettings().loggerStrategy = new WriteStrategy<DenseAction, DenseState>(
        fm.addPath("NlsFinal.log"),
        WriteStrategy<DenseAction, DenseState>::TRANS,
        true /*delete file*/
    );
    for (int n = 0; n < 100; ++n)
        core.runTestEpisode();
    //---

    return 0;
}
