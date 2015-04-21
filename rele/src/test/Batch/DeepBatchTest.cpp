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

#include "DifferentiableNormals.h"
#include "Core.h"
#include "PolicyEvalAgent.h"
#include "parametric/differentiable/GibbsPolicy.h"
#include "policy_search/offpolicy/OffPolicyREINFORCE.h"
#include "BasisFunctions.h"
#include "basis/PolynomialFunction.h"
#include "basis/ConditionBasedFunction.h"
#include "RandomGenerator.h"
#include "FileManager.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <random>
#include <cmath>
#include "DeepSeaTreasure.h"

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
    FileManager fm("Offpolicy", "deep");
    fm.createDir();
    fm.cleanDir();

    DeepSeaTreasure mdp;
    vector<FiniteAction> actions;
    for (int i = 0; i < mdp.getSettings().finiteActionDim; ++i)
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

    for (int i = 0; i < actions.size() -1; ++i)
    {
        bfs.push_back(new AndConditionBasisFunction(pf0,2,i));
        bfs.push_back(new AndConditionBasisFunction(pfs1,2,i));
        bfs.push_back(new AndConditionBasisFunction(pfs2,2,i));
        bfs.push_back(new AndConditionBasisFunction(pfs1s2,2,i));
        bfs.push_back(new AndConditionBasisFunction(d2si,2,i));
        bfs.push_back(new AndConditionBasisFunction(dsi,2,i));
    }

    DenseFeatures basis(bfs);

    ParametricGibbsPolicy<DenseState> behavioral(actions, basis, 1);

    ParametricGibbsPolicy<DenseState> target(actions, basis, 1);
    //---

    PolicyEvalAgent<FiniteAction,DenseState> agent(behavioral);

    ReLe::Core<FiniteAction, DenseState> oncore(mdp, agent);
    CollectorStrategy<FiniteAction, DenseState>* strat = new CollectorStrategy<FiniteAction, DenseState>();
    oncore.getSettings().loggerStrategy = strat;

    int horiz = mdp.getSettings().horizon;
    oncore.getSettings().episodeLenght = horiz;

    int nbTrajectories = 1e4;
    for (int n = 0; n < nbTrajectories; ++n)
        oncore.runTestEpisode();

    Dataset<FiniteAction, DenseState>& data = strat->data;
    ofstream out(fm.addPath("Dataset.csv"), ios_base::out);
    if (out.is_open())
        data.writeToStream(out);
    out.close();

    cout << "# Ended data collection" << endl;


    AdaptiveStep stepl(0.1);

    OffpolicyREINFORCE<FiniteAction, DenseState> offagent(target, behavioral, data.size(), stepl);
    BatchCore<FiniteAction, DenseState> offcore(mdp, offagent, data);
    offcore.getSettings().loggerStrategy = new WriteStrategy<FiniteAction, DenseState>(
        fm.addPath("deep.log"),
        WriteStrategy<FiniteAction, DenseState>::AGENT,
        true /*delete file*/
    );
    offcore.getSettings().episodeLenght = horiz;

    int nbUpdates = 10;
    double every, bevery;
    every = bevery = 0.1; //%
    for (int i = 0; i < nbUpdates; i++)
    {
        offcore.processBatchData();

        int p = 100 * i/static_cast<double>(nbUpdates);
        cout << "### " << p << "% ###" << endl;
        //                cout << dist.getParameters().t();
        offcore.getSettings().testEpisodeN = 100;
        arma::vec J = offcore.runBatchTest();
        cout << "mean score: " << J(0) << endl;
        every += bevery;
    }

    delete strat;
    delete pf0;
    delete pfs1;
    delete pfs2;
    delete pfs1s2;
    delete d2si;
    delete dsi;
    return 0;
}
