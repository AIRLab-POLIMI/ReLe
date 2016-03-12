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

#include "rele/environments/TaxiFuel.h"

#include "rele/core/Core.h"

#include "rele/algorithms/policy_search/gradient/GPOMDPAlgorithm.h"

#include "rele/policy/parametric/differentiable/GenericGibbsPolicy.h"
#include "rele/approximators/features/DenseFeatures.h"
#include "rele/approximators/basis/IdentityBasis.h"
#include "rele/approximators/basis/PolynomialFunction.h"
#include "rele/approximators/basis/GaussianRbf.h"
#include "rele/approximators/basis/ConditionBasedFunction.h"

#include "rele/utils/FileManager.h"
#include "rele/utils/ConsoleManager.h"

#include <iostream>

using namespace std;
using namespace ReLe;

int main(int argc, char *argv[])
{
    TaxiFuel taxiMDP;

    FileManager fm("TaxiFuel", "PG");
    fm.createDir();
    fm.cleanDir();


    vector<FiniteAction> actions = FiniteAction::generate(TaxiFuel::ACTIONNUMBER);

    //-- Features
    //BasisFunctions basis = GaussianRbf::generate({5, 5, 3, 3}, {0, 5, 0, 5, 0, 12, -1, 1});
    //BasisFunctions basis = PolynomialFunction::generate(1, TaxiFuel::STATESIZE);
    //BasisFunctions basis = IdentityBasis::generate(TaxiFuel::STATESIZE);
    BasisFunctions basisSpace = VectorFiniteIdentityBasis::generate(2, 5);

    vector<unsigned int> indexes;
    vector<unsigned int> values;
    indexes.push_back(TaxiFuel::onBoard);
    values.push_back(2);
    indexes.push_back(TaxiFuel::location);
    values.push_back(4);
    indexes.push_back(TaxiFuel::destination);
    values.push_back(4);
    BasisFunctions basis = AndConditionBasisFunction::generate(basisSpace, indexes, values);
    basis.push_back(new IdentityBasis(TaxiFuel::fuel));

    BasisFunctions basisGibbs = AndConditionBasisFunction::generate(basis, TaxiFuel::STATESIZE, actions.size()-1);
    DenseFeatures phi(basisGibbs);
    cout << phi.rows() << endl;
    LinearApproximator reg(phi);

    double temperature = 100;
    GenericParametricGibbsPolicy<DenseState> policy(actions, reg, temperature);

    //-- agent
    int nbepperpol = 10, nbstep = 100;
    //AdaptiveStep stepRule(0.01);
    ConstantGradientStep stepRule(0.01);
    GPOMDPAlgorithm<FiniteAction, DenseState> agent(policy, nbepperpol, nbstep, stepRule,
            GPOMDPAlgorithm<FiniteAction, DenseState>::BaseLineType::MULTI);

    Core<FiniteAction, DenseState> core(taxiMDP, agent);
    //--


    int episodes = 50000;
    core.getSettings().episodeLength = 100;
    core.getSettings().loggerStrategy = new WriteStrategy<FiniteAction, DenseState>(fm.addPath("TaxiFuel.log"),
            WriteStrategy<FiniteAction, DenseState>::AGENT);


    ConsoleManager console(episodes, 1);
    console.printInfo("starting learning");
    for (int i = 0; i < episodes; i++)
    {
        console.printProgress(i);
        core.runEpisode();
        if(temperature > 0.1)
        {
            temperature *= 0.9999;
            cout << "temperature: " << temperature << endl;
            policy.setTemperature(temperature);
        }
    }

    delete core.getSettings().loggerStrategy;

    console.printInfo("Starting evaluation episode");
    core.getSettings().loggerStrategy = new WriteStrategy<FiniteAction, DenseState>(fm.addPath("TaxiFuel.log"),
            WriteStrategy<FiniteAction, DenseState>::TRANS);

    for(int i = 0; i < 10; i++)
    {
        cout << "# " << i + 1 << "/10" << endl;
        core.runTestEpisode();
    }

    delete core.getSettings().loggerStrategy;

    cout << "p" << policy.getParameters().t();
    return 0;
}
