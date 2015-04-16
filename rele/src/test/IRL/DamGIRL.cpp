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

#include "Core.h"
#include "parametric/differentiable/GibbsPolicy.h"
#include "BasisFunctions.h"
#include "parametric/differentiable/NormalPolicy.h"
#include "basis/PolynomialFunction.h"
#include "basis/GaussianRBF.h"
#include "basis/ConditionBasedFunction.h"

#include "Dam.h"
#include "PolicyEvalAgent.h"
#include "algorithms/GIRL.h"
#include "policy_search/PGPE/PGPE.h"
#include "ParametricRewardMDP.h"
#include "RandomGenerator.h"
#include "FileManager.h"

#include "policy_search/onpolicy/FunctionGradient.h"
#include "policy_search/onpolicy/PolicyGradientAlgorithm.h"

using namespace std;
using namespace ReLe;
using namespace arma;

class dam_IRL_Reward : public IRLParametricReward<DenseAction, DenseState>,
        public RewardTransformation
{
public:

    dam_IRL_Reward(Dam& mdp)
        : mdp(mdp)
    {
        weights.zeros(2);
    }

    double operator()(DenseState& s, DenseAction& a, DenseState& ns)
    {
        Reward r(mdp.getSettings().rewardDim);
        DenseState nexts(1);
        mdp.setCurrentState(s);
        mdp.step(a, nexts, r);

        double val = 0.0;
        for (int i = 0; i < weights.n_elem; ++i)
            val += weights[i] * r[i];
        return val;
    }

    double operator()(const Reward& r)
    {
        double val = 0.0;
        for (int i = 0; i < weights.n_elem; ++i)
            val += weights[i] * r[i];
        return val;
    }

    arma::mat diff(DenseState& s, DenseAction& a, DenseState& ns)
    {
        arma::mat m(1,weights.n_elem);
        Reward r(mdp.getSettings().rewardDim);
        DenseState nexts(1);
        mdp.setCurrentState(s);
        mdp.step(a, nexts, r);

        for (int i = 0; i < weights.n_elem; ++i)
            m(0,i) = r[i];
        return m;
    }

private:

    Dam& mdp;
};

void help()
{
    cout << "dam_GIRL [algorithm]" << endl;
    cout << " - algorithm: r, rb, g, gb (default)" << endl;
}


int main(int argc, char *argv[])
{
    //    RandomGenerator::seed(12354);

    /*** check inputs ***/
    char alg[10];
    GIRL<DenseAction,DenseState>::AlgType atype;
    if (argc > 1)
    {
        if (strcmp(argv[1], "r") == 0)
        {
            cout << "GIRL REINFORCE" << endl;
            atype = GIRL<DenseAction,DenseState>::AlgType::R;
        }
        else if (strcmp(argv[1], "rb") == 0)
        {
            cout << "GIRL REINFORCE BASE" << endl;
            atype = GIRL<DenseAction,DenseState>::AlgType::RB;
        }
        else if (strcmp(argv[1], "g") == 0)
        {
            cout << "GIRL GPOMDP" << endl;
            atype = GIRL<DenseAction,DenseState>::AlgType::G;
        }
        else if (strcmp(argv[1], "gb") == 0)
        {
            cout << "GIRL GPOMDP BASE" << endl;
            atype = GIRL<DenseAction,DenseState>::AlgType::GB;
        }
        else
        {
            std::cout << "Error unknown argument " << argv[1] << std::endl;
            help();
            exit(1);
        }
        strcpy(alg, argv[1]);
    }
    else
    {
        atype = GIRL<DenseAction,DenseState>::AlgType::GB;
        strcpy(alg, "gb");
    }
    /******/

    FileManager fm("dam", "GIRL");
    fm.createDir();
    //    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);

    /*** Set up MDP ***/
    Dam mdp;

    PolynomialFunction *pf = new PolynomialFunction(1,0);
    GaussianRbf* gf1 = new GaussianRbf(0,50);
    GaussianRbf* gf2 = new GaussianRbf(50,20);
    GaussianRbf* gf3 = new GaussianRbf(120,40);
    GaussianRbf* gf4 = new GaussianRbf(160,50);
    DenseBasisVector basis;
    basis.push_back(pf);
    basis.push_back(gf1);
    basis.push_back(gf2);
    basis.push_back(gf3);
    basis.push_back(gf4);
    cout << basis << endl;
    LinearApproximator regressor(mdp.getSettings().continuosStateDim, basis);
    vec p(5);
    p(0) = 50;
    p(1) = -50;
    p(2) = 0;
    p(3) = 0;
    p(4) = 50;
    regressor.setParameters(p);
    MVNLogisticPolicy expertPolicy(&regressor, 50);
    //---

    vec eReward(2);
    eReward(0) = 0.1;
    eReward(1) = 0.9;
#if 1
    /*** learn the optimal policy ***/
    int nbepperpol = 150;
    AdaptiveStep srule(0.001);
    WeightedSumRT rewardtr(eReward);
    GPOMDPAlgorithm<DenseAction, DenseState> agent(expertPolicy, nbepperpol,
                                                   mdp.getSettings().horizon, srule, rewardtr);
    ReLe::Core<DenseAction, DenseState> core(mdp, agent);
    core.getSettings().loggerStrategy = new WriteStrategy<DenseAction, DenseState>(
                fm.addPath("gradient_log_learning.log"),
                WriteStrategy<DenseAction, DenseState>::AGENT,
                true /*delete file*/
                );

    int horiz = mdp.getSettings().horizon;
    core.getSettings().episodeLenght = horiz;

    int nbUpdates = 550;
    int episodes  = nbUpdates*nbepperpol;
    double every, bevery;
    every = bevery = 0.05; //%
    int updateCount = 0;
    for (int i = 0; i < episodes; i++)
    {
        if (i % nbepperpol == 0)
        {
            updateCount++;
            if ((updateCount >= nbUpdates*every) || (updateCount == 1))
            {
                int p = std::floor(100 * (updateCount/static_cast<double>(nbUpdates)));
                cout << "### " << p << "% ###" << endl;
                cout << expertPolicy.getParameters().t();
                core.getSettings().testEpisodeN = 1000;
                arma::vec J = core.runBatchTest();
                cout << "mean score: " << J(0) << endl;
                if (updateCount != 1)
                    every += bevery;
            }
        }

        core.runEpisode();
    }


    cout << endl << "### Ended Optimization" << endl;

#else
    arma::vec param(expertPolicy.getParametersSize());
    //parametri ottenuto con [1.0,0.0]
    //    param << -0.0783 << -0.1144 << -0.2249 << -0.3522 <<  0.0056 << -0.0499 <<  0.8050 <<  1.1729 <<  0.0149 << -0.1588 <<  0.4539 <<  0.5678 <<  0.0112 <<  0.0357 << -0.0725 << -0.1477 <<  0.0047 << -0.0005;

    // parametri ottenuto con [0.6,0.4]
    eReward(0) = 0.6;
    eReward(1) = 0.4;
    param <<-0.0949 << -0.1268 << -0.2743 << -0.3641 << -0.0059 << -0.0665 <<  0.7563 <<  1.1532 <<  0.0561 << -0.1504 <<  0.4044 <<  0.5052 << -0.0334 << -0.0349 << -0.1495 << -0.2065 << -0.0091 << -0.0305;
    expertPolicy.setParameters(param);
#endif


    cout << expertPolicy.getParameters().t();


    PolicyEvalAgent<DenseAction, DenseState> expert(expertPolicy);

    /* Generate DAM expert dataset */
    Core<DenseAction, DenseState> expertCore(mdp, expert);
    CollectorStrategy<DenseAction, DenseState> collection;
    expertCore.getSettings().loggerStrategy = &collection;
    expertCore.getSettings().episodeLenght = 50;
    expertCore.getSettings().testEpisodeN = 100;
    expertCore.runTestEpisodes();


    /* Learn weight with GIRL */
    dam_IRL_Reward rewardRegressor(mdp);
    Dataset<DenseAction,DenseState>& data = collection.data;
    GIRL<DenseAction,DenseState> irlAlg(data, expertPolicy, rewardRegressor,
                                        mdp.getSettings().gamma, atype);

    //Run GIRL
    irlAlg.run();
    arma::vec w = irlAlg.getWeights();

    cout << "Computed weights: " << endl << w.t() << endl;

    char filename[100];
    sprintf(filename, "girl_%s.log", alg);
    ofstream outf(fm.addPath(filename), ios_base::app);
    outf << w.t();


    vec grad, grad2, grad3;
    mat dgrad3;
    GradientFromDataWorker<DenseAction,DenseState> gdw(data, expertPolicy, rewardRegressor, mdp.getSettings().gamma);
    if (atype == GIRL<DenseAction,DenseState>::AlgType::R)
    {
        cout << "PG REINFORCE" << endl;
        grad3 = irlAlg.ReinforceGradient(dgrad3);
        rewardRegressor.setParameters(eReward);
        grad = gdw.ReinforceGradient();
        rewardRegressor.setParameters(w);
        grad2 = gdw.ReinforceGradient();
    }
    else if (atype == GIRL<DenseAction,DenseState>::AlgType::RB)
    {
        cout << "PG REINFORCE BASE" << endl;
        grad3 = irlAlg.ReinforceBaseGradient(dgrad3);
        rewardRegressor.setParameters(eReward);
        grad = gdw.ReinforceBaseGradient();
        rewardRegressor.setParameters(w);
        grad2 = gdw.ReinforceBaseGradient();
    }
    else if (atype == GIRL<DenseAction,DenseState>::AlgType::G)
    {
        cout << "PG GPOMDP" << endl;
        grad3 = irlAlg.GpomdpGradient(dgrad3);
        rewardRegressor.setParameters(eReward);
        grad = gdw.GpomdpGradient();
        rewardRegressor.setParameters(w);
        grad2 = gdw.GpomdpGradient();
    }
    else if (atype == GIRL<DenseAction,DenseState>::AlgType::GB)
    {
        cout << "PG GPOMDP BASE" << endl;
        grad3 = irlAlg.GpomdpBaseGradient(dgrad3);
        rewardRegressor.setParameters(eReward);
        grad = gdw.GpomdpBaseGradient();
        rewardRegressor.setParameters(w);
        grad2 = gdw.GpomdpBaseGradient();
    }
    cout << "Gradient Original (" << eReward[0] << ", " << eReward[1] << "): " << endl << grad.t();
    cout << "\tnorm2: " << norm(grad,2) << endl << endl;
    cout << "Gradient weights IRL (" << w[0] << ", " << w[1] << "): " << endl << grad2.t();
    cout << "\tnorm2: " << norm(grad2,2) << endl << endl;
    cout << "Gradient computed by objective function: " << endl << grad3.t();
    cout << "\tnorm2: " << norm(grad3,2) << endl;
    cout << "Objective function (0.5*nomr(g,2)^2: " << norm(grad3,2)*norm(grad3,2)*0.5 << endl;

    return 0;
}
