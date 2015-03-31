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
#include "DifferentiableNormals.h"
#include "Core.h"
#include "parametric/differentiable/PortfolioNormalPolicy.h"
#include "BasisFunctions.h"
#include "basis/PolynomialFunction.h"
#include "RandomGenerator.h"
#include "FileManager.h"
#include "Portfolio.h"

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
    FileManager fm("Portfolio", "PG");
    fm.createDir();
    fm.cleanDir();

    Portfolio mdp;
    //with these settings
    //max in ( many optimal points ) -> J = 8.5
    //note that there are multiple optimal solutions
    //e.g.
    //10, 10, 10, 10, .... (guardare documenti)

    int dim = mdp.getSettings().continuosStateDim;

    //--- define policy (low level)
    DenseBasisVector basis;
    basis.generatePolynomialBasisFunctions(1,dim);
    delete basis.at(0);
    basis.erase(basis.begin());
    cout << "--- Regressor ---" << endl;
    cout << basis << endl;
    LinearApproximator meanRegressor(dim, basis);


    double epsilon = 0.05;
    PortfolioNormalPolicy policy(epsilon, &meanRegressor);
    //---

    int nbepperpol = 100;
    bool usebaseline = true;
    PolicyGradientAlgorithm<FiniteAction, DenseState> agent(policy, nbepperpol, 1.0, usebaseline);

    ReLe::Core<FiniteAction, DenseState> core(mdp, agent);
    core.getSettings().loggerStrategy = new WriteStrategy<FiniteAction, DenseState>(
        fm.addPath("Portfolio.log"),
        WriteStrategy<FiniteAction, DenseState>::AGENT,
        true /*delete file*/
    );

    int horiz = mdp.getSettings().horizon;
    core.getSettings().episodeLenght = horiz;

    int nbUpdates = 1000;
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
    core.getSettings().loggerStrategy = new WriteStrategy<FiniteAction, DenseState>(
        fm.addPath("PortfolioFinal.log"),
        WriteStrategy<FiniteAction, DenseState>::TRANS,
        true /*delete file*/
    );
    for (int n = 0; n < 100; ++n)
        core.runTestEpisode();
    //---

    return 0;
}
