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

#include "rele/policy/parametric/differentiable/GenericGibbsPolicy.h"
#include "rele/approximators/BasisFunctions.h"
#include "rele/approximators/basis/PolynomialFunction.h"
#include "rele/approximators/basis/ConditionBasedFunction.h"
#include "rele/approximators/features/DenseFeatures.h"

#include "rele/environments/DeepSeaTreasure.h"
#include "PGTest.h"

using namespace std;
using namespace ReLe;
using namespace arma;

/////////////////////////////////////////////////////////////

class deep_2state_identity: public BasisFunction
{
    double operator()(const arma::vec& input) override
    {
        return ((input[0] == 1) && (input[1] == 1))?1:0;
    }
    void writeOnStream(std::ostream& out) override
    {
        out << "deep_2state" << endl;
    }
    void readFromStream(std::istream& in) override {}
};

class deep_state_identity: public BasisFunction
{
    double operator()(const arma::vec& input) override
    {
        return (input[0] == 1)?1:0;
    }
    void writeOnStream(std::ostream& out) override
    {
        out << "deep_state" << endl;
    }
    void readFromStream(std::istream& in) override {}
};

int main(int argc, char *argv[])
{
    CommandLineParser clp;
    gradConfig config = clp.getConfig(argc, argv);
    config.envName = "deep";

    DeepSeaTreasure mdp;
    vector<FiniteAction> actions;
    for (unsigned int i = 0; i < mdp.getSettings().actionsNumber; ++i)
        actions.push_back(FiniteAction(i));

    //--- policy setup
    PolynomialFunction* pf0 = new PolynomialFunction();
    vector<unsigned int> dim = {0,1};
    vector<unsigned int> deg = {1,0};
    PolynomialFunction* pfs1 = new PolynomialFunction(dim,deg);
    deg = {0,1};
    PolynomialFunction* pfs2 = new PolynomialFunction(dim,deg);
    deg = {1,1};
    PolynomialFunction* pfs1s2 = new PolynomialFunction(dim, deg);
    deep_2state_identity* d2si = new deep_2state_identity();
    deep_state_identity* dsi = new deep_state_identity();

    BasisFunctions bfs;

    for (int i = 0; i < actions.size() - 1; ++i)
    {
        bfs.push_back(new AndConditionBasisFunction(pf0,2,i));
        bfs.push_back(new AndConditionBasisFunction(pfs1,2,i));
        bfs.push_back(new AndConditionBasisFunction(pfs2,2,i));
        bfs.push_back(new AndConditionBasisFunction(pfs1s2,2,i));
        bfs.push_back(new AndConditionBasisFunction(d2si,2,i));
        bfs.push_back(new AndConditionBasisFunction(dsi,2,i));
    }

    DenseFeatures phi(bfs);
    LinearApproximator reg(phi);

    GenericParametricGibbsPolicy<DenseState> policy(actions, reg, 1);

    PGTest<FiniteAction, DenseState> pgTest(config, mdp, policy);
    pgTest.run();

    delete pf0;
    delete pfs1;
    delete pfs2;
    delete pfs1s2;
    delete d2si;
    delete dsi;
    return 0;
}
