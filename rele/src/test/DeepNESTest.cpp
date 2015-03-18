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
#include "parametric/differentiable/GibbsPolicy.h"
#include "BasisFunctions.h"
#include "basis/PolynomialFunction.h"
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
    PolynomialFunction* pfs1s2 = new PolynomialFunction(2,1);
    deep_2state_identity* d2si = new deep_2state_identity();
    deep_state_identity* dsi = new deep_state_identity();

    DenseBasisVector basis;
    for (int i = 0; i < actions.size() -1; ++i)
    {
        basis.push_back(pf0);
        basis.push_back(pfs1);
        basis.push_back(pfs2);
        basis.push_back(pfs1s2);
        basis.push_back(d2si);
        basis.push_back(dsi);
    }
    cout << basis << endl;
    cout << "basis length: " << basis.size() << endl;
    LinearApproximator regressor(mdp.getSettings().continuosStateDim, basis);
    ParametricGibbsPolicy<DenseState> policy(actions, &regressor);
    //---

    int nparams = basis.size();
    arma::vec mean(nparams, fill::zeros);
    arma::mat cov(nparams, nparams, arma::fill::eye);
    cov *= 10;
    mat cholMtx = chol(cov);
    ParametricCholeskyNormal dist(mean.n_elem, mean, cholMtx);




    int nbepperpol = 1, nbpolperupd = 50;
    bool usebaseline = true;
//    PGPE<FiniteAction, DenseState> agent(dist, policy, nbepperpol, nbpolperupd, 0.1, usebaseline);
//    agent.setNormalization(true);
    xNES<FiniteAction, DenseState, ParametricCholeskyNormal> agent(dist, policy, nbepperpol, nbpolperupd, 0.1, usebaseline);

    const std::string outfile = "lqrNesAgentOut.txt";
    ReLe::Core<FiniteAction, DenseState> core(mdp, agent);
    core.getSettings().loggerStrategy = new WriteStrategy<FiniteAction, DenseState>(
        outfile, WriteStrategy<FiniteAction, DenseState>::AGENT
    );
    //--- delete file
    std::ofstream ofs(outfile, std::ios_base::out);
    ofs.close();
    //---

    core.getSettings().episodeLenght = mdp.getSettings().horizon;

    int nbUpdates = 100;
    int episodes  = nbUpdates*nbepperpol*nbpolperupd;
    double every, bevery;
    every = bevery = 0.01; //%
    int updateCount = 0;
    for (int i = 0; i < episodes; i++)
    {
        //        cout << "starting episode" << endl;
        core.runEpisode();

        int v = nbepperpol*nbpolperupd;
        if (i % v == 0)
        {
            updateCount++;
            if (updateCount >= nbUpdates*every)
            {
                int p = 100 * updateCount/static_cast<double>(nbUpdates);
//                cout << "### " << p << "% ###" << endl;
//                cout << dist.getParameters().t();
                arma::vec J = core.runBatchTest(100);
                cout << "mean score: " << J(0) << endl;
                every += bevery;
            }
        }
    }


    cout << core.runBatchTest(10) << endl;

    return 0;
}
