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


//class ConditionBasisFunction: public BasisFunction {
//public:
//    ConditionBasisFunction(BasisFunction* bfs, int condition_val)
//        :bfs(bfs), val(condition_val)
//    {
//    }

//    double operator()(const arma::vec& input)
//    {
//        int tmp = (input[2] == val)?1:0;
//        if (tmp == 0)
//        {
//            return 0;
//        }
//        return (*bfs)(input);
//    }
//    void writeOnStream(std::ostream& out)
//    {
//        out << "condition_bfs " << val << endl;
//    }
//    void readFromStream(std::istream& in) {}

//    private:
//        BasisFunction* bfs;
//        int val;
//};

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

    //--- distribution setup
    //----- ParametricCholeskyNormal
    int nparams = basis.size();
    arma::vec mean(nparams, fill::zeros);
    arma::mat cov(nparams, nparams, arma::fill::eye);
    mat cholMtx = chol(cov);
    ParametricCholeskyNormal dist(mean.n_elem, mean, cholMtx);
    //----- ParametricDiagonalNormal
    //    vec mean(nparams, fill::zeros);
    //    vec sigmas(nparams, fill::ones);
    //    ParametricDiagonalNormal dist(mean, sigmas);
    //-----
    //---


    int nbepperpol = 1, nbpolperupd = 300;
    bool usebaseline = true;
    //    PGPE<FiniteAction, DenseState> agent(dist, policy, nbepperpol, nbpolperupd, 0.1, usebaseline);
    //    agent.setNormalization(true);
    NES<FiniteAction, DenseState> agent(dist, policy, nbepperpol, nbpolperupd, 0.1, usebaseline);
    //    xNES<FiniteAction, DenseState, ParametricCholeskyNormal> agent(dist, policy, nbepperpol, nbpolperupd, 0.1, usebaseline);

    const std::string outfile = "deep_bbo_out.txt";
    ReLe::Core<FiniteAction, DenseState> core(mdp, agent);
    core.getSettings().loggerStrategy = new WriteStrategy<FiniteAction, DenseState>(
                outfile, WriteStrategy<FiniteAction, DenseState>::AGENT,
                true /*delete file*/
                );

    int horiz = mdp.getSettings().horizon;
    core.getSettings().episodeLenght = horiz;

    int nbUpdates = 50;
    int episodes  = nbUpdates*nbepperpol*nbpolperupd;
    double every, bevery;
    every = bevery = 0.01; //%
    int updateCount = 0;
    for (int i = 0; i < episodes; i++)
    {
        core.runEpisode();

        int v = nbepperpol*nbpolperupd;
        if (i % v == 0)
        {
            updateCount++;
            if (updateCount >= nbUpdates*every)
            {
                int p = 100 * updateCount/static_cast<double>(nbUpdates);
                cout << "### " << p << "% ###" << endl;
                //                cout << dist.getParameters().t();
                arma::vec J = core.runBatchTest(100);
                cout << "mean score: " << J(0) << endl;
                every += bevery;
            }
        }
    }


    int nbTestEpisodes = 1000;
    cout << "Final test [#episodes: " << nbTestEpisodes << " ]" << endl;
    cout << core.runBatchTest(1000) << endl;

    //--- collect some trajectories
    const std::string testOutfile = "deep_bbo_final_trajectories.csv";
    core.getSettings().loggerStrategy = new WriteStrategy<FiniteAction, DenseState>(
                    testOutfile, WriteStrategy<FiniteAction, DenseState>::TRANS,
                    true /*delete file*/
                    );
    core.runTestEpisode();
    //---

    return 0;
}
