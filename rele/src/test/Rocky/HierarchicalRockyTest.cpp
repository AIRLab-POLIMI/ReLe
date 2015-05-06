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

#include "Rocky.h"

#include "Core.h"

#include "policy_search/gradient/hierarchical/HierarchicalGPOMDP.h"

#include "parametric/differentiable/GibbsPolicy.h"
#include "features/DenseFeatures.h"
#include "basis/IdentityBasis.h"
#include "basis/SubspaceBasis.h"
#include "basis/GaussianRbf.h"

#include "FileManager.h"
#include "ConsoleManager.h"

#include "RockyOptions.h"

#include <iostream>

using namespace std;
using namespace ReLe;
using namespace arma;


int main(int argc, char *argv[])
{
    FileManager fm("Rocky", "HPG");
    fm.createDir();
    fm.cleanDir();

    Rocky rocky;


    //-- Features
    BasisFunctions radialBasis1 = GaussianRbf::generate(8,
    {
        -10, 10,
        -10, 10,
        -M_PI, M_PI
    });

    BasisFunctions radialBasis2 = GaussianRbf::generate(8,
    {
        -10, 10,
        -10, 10,
        -M_PI, M_PI
    });

    BasisFunctions basis;
    BasisFunctions basis1 = SubspaceBasis::generate(radialBasis1, span(Rocky::x, Rocky::theta));
    BasisFunctions basis2 = SubspaceBasis::generate(radialBasis2, span(Rocky::xr, Rocky::thetar));
    basis.insert(basis.end(), basis1.begin(), basis1.end());
    basis.insert(basis.end(), basis2.begin(), basis2.end());
    basis.push_back(new IdentityBasis(Rocky::energy));
    basis.push_back(new IdentityBasis(Rocky::food));

    DenseFeatures phi(basis);
    cout << "Features size: " << phi.rows() << endl;


    //-- options
    Eat eat;
    Home home;
    Feed feed;
    Escape1 escape1;
    Escape2 escape2;
    Escape3 escape3;

    vector<Option<DenseAction, DenseState>*> options;
    options.push_back(&eat);
    options.push_back(&home);
    options.push_back(&feed);
    options.push_back(&escape1);
    options.push_back(&escape2);
    options.push_back(&escape3);

    vector<FiniteAction> actions;
    for(int i = 0; i < options.size(); ++i)
        actions.push_back(FiniteAction(i));

    ParametricGibbsPolicy<DenseState> rootPolicyOption(actions, phi, 1);
    DifferentiableOption<DenseAction, DenseState> rootOption(rootPolicyOption, options);
    //--

    //-- agent
    int nbepperpol = 5, nbstep = 10000;
    AdaptiveStep stepRule(0.01);
    HierarchicalGPOMDPAlgorithm<DenseAction, DenseState> agent(rootOption, nbepperpol, nbstep, stepRule,
            HierarchicalGPOMDPAlgorithm<DenseAction, DenseState>::MULTI);

    Core<DenseAction, DenseState> core(rocky, agent);
    //--


    int episodes = 3000;
    core.getSettings().episodeLenght = 10000;
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

    return 0;

}
