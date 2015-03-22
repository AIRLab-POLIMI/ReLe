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

#include "DeepSeaTreasure.h"
#include "policy_search/NES/NES.h"
#include "DifferentiableNormals.h"
#include "Core.h"
#include "PolicyEvalAgent.h"
#include "parametric/differentiable/GibbsPolicy.h"
#include "BasisFunctions.h"
#include "basis/PolynomialFunction.h"
#include "basis/ConditionBasedFunction.h"
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


class deep_2state_identity: public BasisFunction
{
    double operator()(const arma::vec& input)
    {
        return ((input[0] == 1) && (input[1] == 1))?1:0;
    }
    void writeOnStream(std::ostream& out)
    {
        out << "deep_2state" << endl;
    }
    void readFromStream(std::istream& in) {}
};

class deep_state_identity: public BasisFunction
{
    double operator()(const arma::vec& input)
    {
        return (input[0] == 1)?1:0;
    }
    void writeOnStream(std::ostream& out)
    {
        out << "deep_state" << endl;
    }
    void readFromStream(std::istream& in) {}
};

int main(int argc, char *argv[])
{
    DeepSeaTreasure mdp;
    vector<FiniteAction> actions;
    for (int i = 0; i < mdp.getSettings().finiteActionDim; ++i)
        actions.push_back(FiniteAction(i));

    //--- policy setup
    PolynomialFunction* pf0 = new PolynomialFunction(2,0);
    vector<unsigned int> dim = {0,1};
    vector<unsigned int> deg = {1,0};
    PolynomialFunction* pfs1 = new PolynomialFunction(dim,deg);
    deg = {0,1};
    PolynomialFunction* pfs2 = new PolynomialFunction(dim,deg);
    deg = {1,1};
    PolynomialFunction* pfs1s2 = new PolynomialFunction(dim, deg);
    deep_2state_identity* d2si = new deep_2state_identity();
    deep_state_identity* dsi = new deep_state_identity();

    DenseBasisVector basis;
    for (int i = 0; i < actions.size() -1; ++i)
    {
        basis.push_back(new AndConditionBasisFunction(pf0,2,i));
        basis.push_back(new AndConditionBasisFunction(pfs1,2,i));
        basis.push_back(new AndConditionBasisFunction(pfs2,2,i));
        basis.push_back(new AndConditionBasisFunction(pfs1s2,2,i));
        basis.push_back(new AndConditionBasisFunction(d2si,2,i));
        basis.push_back(new AndConditionBasisFunction(dsi,2,i));

        //        basis.push_back(new ConditionBasisFunction(pf0,i));
        //        basis.push_back(new ConditionBasisFunction(pfs1,i));
        //        basis.push_back(new ConditionBasisFunction(pfs2,i));
        //        basis.push_back(new ConditionBasisFunction(pfs1s2,i));
        //        basis.push_back(new ConditionBasisFunction(d2si,i));
        //        basis.push_back(new ConditionBasisFunction(dsi,i));
    }
    //    cout << basis << endl;
    //    cout << "basis length: " << basis.size() << endl;

    LinearApproximator regressor(mdp.getSettings().continuosStateDim + 1, basis);
    ParametricGibbsPolicy<DenseState> policy(actions, &regressor, 1e8);
    //---

    PolicyEvalAgent
    <FiniteAction,DenseState,ParametricGibbsPolicy<DenseState> >
    agent(policy);

    const std::string outfile = "deep_bbo_out.txt";
    ReLe::Core<FiniteAction, DenseState> core(mdp, agent);
    CollectorStrategy<FiniteAction, DenseState>* strat = new CollectorStrategy<FiniteAction, DenseState>();
    core.getSettings().loggerStrategy = strat;

    int horiz = mdp.getSettings().horizon;
    core.getSettings().episodeLenght = horiz;

    for (int n = 0; n < 10; ++n)
        core.runTestEpisode();

    TrajectoryData<FiniteAction, DenseState>& data = strat->data;
    data.WriteToStream(cout);

    return 0;
}
