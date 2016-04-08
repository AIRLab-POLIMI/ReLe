/*
 * rele,
 *
 *
 * Copyright (C) 2015 Davide Tateo & Matteo Pirotta
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

#include "rele/environments/Rocky.h"

#include "rele/algorithms/policy_search/REPS/REPS.h"
#include "rele/statistics/DifferentiableNormals.h"
#include "rele/core/Core.h"
//#include "rele/policy/parametric/differentiable/LinearPolicy.h"
#include "RockyPolicy.h"
#include "rele/approximators/BasisFunctions.h"

#include "rele/utils/FileManager.h"
#include "rele/utils/ConsoleManager.h"

#include <iostream>

using namespace std;
using namespace ReLe;
using namespace arma;


int main(int argc, char *argv[])
{
    FileManager fm("Rocky", "REPS");
    fm.createDir();
    fm.cleanDir();

    Rocky rocky;

    //-- Low level policy
    int dim = rocky.getSettings().stateDimensionality;
    int actionDim = rocky.getSettings().actionDimensionality;

    RockyPolicy policy(0.01);
    //--

    //-- high level policy
    int dimPolicy = policy.getParametersSize();
    arma::vec mean(dimPolicy, fill::zeros);
    arma::mat cov(dimPolicy, dimPolicy, fill::eye);

    cov *= 5;


    ParametricFullNormal dist(mean, cov);
    //--

    //-- REPS agent
    int nbepperpol = 1, nbpolperupd = 50;
    REPS<DenseAction, DenseState> agent(dist,policy,nbepperpol,nbpolperupd);
    agent.setEps(0.5);

    Core<DenseAction, DenseState> core(rocky, agent);
    //--


    int episodes = nbepperpol*nbpolperupd*60;
    core.getSettings().episodeLength = 10000;
    core.getSettings().loggerStrategy = new WriteStrategy<DenseAction, DenseState>(fm.addPath("Rocky.log"),
            WriteStrategy<DenseAction, DenseState>::outType::AGENT);


    ConsoleManager console(episodes, 1);
    console.printInfo("starting learning");
    for (int i = 0; i < episodes; i++)
    {
        console.printProgress(i);
        core.runEpisode();
    }

    delete core.getSettings().loggerStrategy;

    console.printInfo("Starting evaluation episode");
    core.getSettings().loggerStrategy = new WriteStrategy<DenseAction, DenseState>(fm.addPath("Rocky.log"),
            WriteStrategy<DenseAction, DenseState>::outType::TRANS);
    core.runTestEpisode();

    delete core.getSettings().loggerStrategy;

    cout << "Parameters" << endl;
    cout << setprecision(OS_PRECISION);
    cout << dist.getParameters() << endl;

    return 0;

}
