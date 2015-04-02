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

#include "policy_search/onpolicy/PolicyGradientAlgorithm.h"
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
#include "../../include/rele/environments/LQR.h"

using namespace std;
using namespace ReLe;
using namespace arma;

int main(int argc, char *argv[])
{
    FileManager fm("LQR", "PG");
    fm.createDir();
    fm.cleanDir();

    LQR mdp(1,1); //with these settings the optimal value is -0.6180 (for the linear policy)

    PolynomialFunction* pf = new PolynomialFunction(1,1);
    cout << *pf << endl;
    DenseBasisVector basis;
    basis.push_back(pf);
    cout << basis << endl;
    LinearApproximator regressor(mdp.getSettings().continuosStateDim, basis);
    NormalPolicy policy(0.1, &regressor);

    int nbepperpol = 2;
    unsigned int rewardId = 0;
    bool usebaseline = true;
    GPOMDPAlgorithm<DenseAction, DenseState> agent(policy, nbepperpol,
            mdp.getSettings().horizon, 0.1, usebaseline, rewardId);
//    REINFORCEAlgorithm<DenseAction, DenseState> agent(policy, nbepperpol,
//                                                   0.1, usebaseline, rewardId);

    ReLe::Core<DenseAction, DenseState> core(mdp, agent);
    core.getSettings().loggerStrategy = new WriteStrategy<DenseAction, DenseState>(
        fm.addPath("LQR.log"),
        WriteStrategy<DenseAction, DenseState>::ALL,
        true /*delete file*/
    );

    int horiz = mdp.getSettings().horizon;
    core.getSettings().episodeLenght = horiz;

    int nbUpdates = 1;
    int episodes  = nbUpdates*nbepperpol;
    double every, bevery;
    every = bevery = 0.1; //%
    int updateCount = 0;
    for (int i = 0; i < episodes; i++)
    {
        core.runEpisode();

        int v = nbepperpol;
        if (i % v == 0)
        {
            updateCount++;
            if ((updateCount >= nbUpdates*every) || (updateCount == 1))
            {
                int p = 100 * updateCount/static_cast<double>(nbUpdates);
                cout << "### " << p << "% ###" << endl;
                cout << policy.getParameters().t();
                core.getSettings().testEpisodeN = 1000;
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
