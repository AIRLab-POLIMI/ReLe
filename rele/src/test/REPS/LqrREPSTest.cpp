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

#include "policy_search/REPS/REPS.h"
#include "DifferentiableNormals.h"
#include "Core.h"
#include "parametric/differentiable/LinearPolicy.h"
#include "BasisFunctions.h"
#include "basis/IdentityBasis.h"

#include "FileManager.h"
#include "ConsoleManager.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <random>
#include <cmath>
#include "LQR.h"

using namespace std;
using namespace ReLe;
using namespace arma;

int main(int argc, char *argv[])
{
    FileManager fm("LQR", "REPS");
    fm.createDir();
    fm.cleanDir();

    LQR mdp(1, 1); //with these settings the optimal value is -0.6180 (for the linear policy)

    arma::vec mean(1);
    mean[0] = -0.1;
    arma::mat cov(1, 1, arma::fill::eye);
    cov *= 0.01;

    ParametricNormal dist(mean, cov);

    IdentityBasis* pf = new IdentityBasis(0);
    DenseFeatures phi(pf);
    DetLinearPolicy<DenseState> policy(phi);

    int nbepperpol = 1, nbpolperupd = 250;
    REPS<DenseAction, DenseState, ParametricNormal> agent(dist,policy,nbepperpol,nbpolperupd);
    agent.setEps(0.5);

    ReLe::Core<DenseAction, DenseState> core(mdp, agent);

    core.getSettings().loggerStrategy = new WriteStrategy<DenseAction,
    DenseState>(fm.addPath("agent.log"),
                WriteStrategy<DenseAction, DenseState>::AGENT);

    int nbUpdates = 800;
    int episodes  = nbUpdates*nbepperpol*nbpolperupd;

    ConsoleManager console(episodes, 1);
    for (int i = 0; i < episodes; i++)
    {
        core.getSettings().episodeLenght = mdp.getSettings().horizon;
        console.printProgress(i);
        core.runEpisode();
    }

    delete core.getSettings().loggerStrategy;

    cout << dist.getMean().t() << endl;
    cout << dist.getCovariance() << endl;

    return 0;
}
