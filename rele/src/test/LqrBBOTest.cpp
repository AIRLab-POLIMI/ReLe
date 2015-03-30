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
#include "parametric/differentiable/LinearPolicy.h"
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
#include "../../include/rele/environments/LQR.h"

using namespace std;
using namespace ReLe;
using namespace arma;

int main(int argc, char *argv[])
{
    FileManager fm("LQR", "BBO");
    fm.createDir();
    fm.cleanDir();

    LQR mdp(1,1); //with these settings the optimal value is -0.6180 (for the linear policy)

    PolynomialFunction* pf = new PolynomialFunction(1,1);
    cout << *pf << endl;
    DenseBasisVector basis;
    basis.push_back(pf);
    cout << basis << endl;
    LinearApproximator regressor(mdp.getSettings().continuosStateDim, basis);
    DetLinearPolicy<DenseState> policy(&regressor);

    //--- distribution setup
    int nparams = basis.size();
    arma::vec mean(nparams, fill::zeros);
    mean[0] = -0.1;

    //----- ParametricNormal
    //    arma::mat cov(nparams, nparams, arma::fill::eye);
    //    cov *= 0.1;
    //    ParametricNormal dist(mean, cov);
    //----- ParametricLogisticNormal
    //    ParametricLogisticNormal dist(mean, zeros(nparams), 1);
    //----- ParametricCholeskyNormal
    //    arma::mat cov(nparams, nparams, arma::fill::eye);
    //    mat cholMtx = chol(cov);
    //    ParametricCholeskyNormal dist(mean, cholMtx);
    //----- ParametricDiagonalNormal
    vec sigmas(nparams, fill::ones);
    ParametricDiagonalNormal dist(mean, sigmas);
    //-----
    //---

    int nbepperpol = 1, nbpolperupd = 100;
    bool usebaseline = false;
    //    PGPE<DenseAction, DenseState> agent(dist, policy, nbepperpol, nbpolperupd, 0.002, usebaseline);
    //    agent.setNormalization(true);
    NES<DenseAction, DenseState> agent(dist, policy, nbepperpol, nbpolperupd, 0.5, usebaseline);
    //        REPS<DenseAction, DenseState, ParametricNormal> agent(dist,policy,nbepperpol,nbpolperupd);
    //        agent.setEps(0.5);

    ReLe::Core<DenseAction, DenseState> core(mdp, agent);
    core.getSettings().loggerStrategy = new WriteStrategy<DenseAction, DenseState>(
        fm.addPath("LQR.log"),
        WriteStrategy<DenseAction, DenseState>::AGENT,
        true /*delete file*/
    );

    int horiz = mdp.getSettings().horizon;
    core.getSettings().episodeLenght = horiz;

    int nbUpdates = 600;
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
        fm.addPath("LQRFinal.log"),
        WriteStrategy<DenseAction, DenseState>::TRANS,
        true /*delete file*/
    );
    for (int n = 0; n < 100; ++n)
        core.runTestEpisode();
    //---
    return 0;
}
