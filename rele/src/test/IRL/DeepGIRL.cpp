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
#include "DifferentiableNormals.h"
#include "basis/PolynomialFunction.h"
#include "basis/ConditionBasedFunction.h"
#include "features/DenseFeatures.h"

#include "DeepSeaTreasure.h"
#include "PolicyEvalAgent.h"
#include "algorithms/GIRL.h"
#include "policy_search/PGPE/PGPE.h"
#include "ParametricRewardMDP.h"
#include "RandomGenerator.h"
#include "FileManager.h"

#include "policy_search/onpolicy/FunctionGradient.h"
#include "policy_search/onpolicy/GPOMDPAlgorithm.h"
#include "policy_search/onpolicy/REINFORCEAlgorithm.h"

using namespace std;
using namespace ReLe;
using namespace arma;

class Deep_IRL_Reward : public IRLParametricReward<FiniteAction, DenseState>,
    public RewardTransformation
{
public:

    Deep_IRL_Reward()
    {
        weights.set_size(2);
    }

    double operator()(DenseState& s, FiniteAction& a, DenseState& ns)
    {
        //        std::cout << deep_reward_treasure(ns) << " -1" << std::endl;
        return weights(0)*deep_reward_treasure(ns) - weights(1);
    }

    double operator()(const Reward& r)
    {
        return weights(0)*r[0] + weights(1)*r[1];
    }

    arma::mat diff(DenseState& s, FiniteAction& a, DenseState& ns)
    {
        arma::mat m(1,2);
        m(0) = deep_reward_treasure(ns);
        m(1) = -1;
        return m;
    }

private:
    double deep_reward_treasure(DenseState& state)
    {
        int xdim = 11;
        int ydim = 10;
        arma::mat reward(xdim+1,ydim+1,arma::fill::zeros);
        reward(2,1) = 1;
        reward(3,2) = 2;
        reward(4,3) = 3;
        reward(5,4) = 5;
        reward(5,5) = 8;
        reward(5,6) = 16;
        reward(8,7) = 24;
        reward(8,8) = 50;
        reward(10,9) = 74;
        reward(11,10) = 124;
        return reward(state[0],state[1]);

    }
};

class DEEP_R1 : public IRLParametricReward<FiniteAction, DenseState>
{
public:
    double operator()(DenseState& s, FiniteAction& a, DenseState& ns)
    {
        return deep_reward_treasure(ns);
    }

    arma::mat diff(DenseState& s, FiniteAction& a, DenseState& ns)
    {
        return arma::mat();
    }
private:
    double deep_reward_treasure(DenseState& state)
    {
        int xdim = 11;
        int ydim = 10;
        arma::mat reward(xdim+1,ydim+1,arma::fill::zeros);
        reward(2,1) = 1;
        reward(3,2) = 2;
        reward(4,3) = 3;
        reward(5,4) = 5;
        reward(5,5) = 8;
        reward(5,6) = 16;
        reward(8,7) = 24;
        reward(8,8) = 50;
        reward(10,9) = 74;
        reward(11,10) = 124;
        return reward(state[0],state[1]);

    }
};
class DEEP_R2 : public IRLParametricReward<FiniteAction, DenseState>
{
public:
    double operator()(DenseState& s, FiniteAction& a, DenseState& ns)
    {
        return -1;
    }

    arma::mat diff(DenseState& s, FiniteAction& a, DenseState& ns)
    {
        return arma::mat();
    }
};


/////////////////////////////////////////////////////////////

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
/////////////////////////////////////////////////////////////

void help()
{
    cout << "deep_GIRL [algorithm]" << endl;
    cout << " - algorithm: r, rb, g, gb (default)" << endl;
}


int main(int argc, char *argv[])
{
//    RandomGenerator::seed(49921158);

    /*** check inputs ***/
    char alg[10];
    IRLGradType atype;
    if (argc > 1)
    {
        if (strcmp(argv[1], "r") == 0)
        {
            cout << "GIRL REINFORCE" << endl;
            atype = IRLGradType::R;
        }
        else if (strcmp(argv[1], "rb") == 0)
        {
            cout << "GIRL REINFORCE BASE" << endl;
            atype = IRLGradType::RB;
        }
        else if (strcmp(argv[1], "g") == 0)
        {
            cout << "GIRL GPOMDP" << endl;
            atype = IRLGradType::G;
        }
        else if (strcmp(argv[1], "gb") == 0)
        {
            cout << "GIRL GPOMDP BASE" << endl;
            atype = IRLGradType::GB;
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
        atype = IRLGradType::GB;
        strcpy(alg, "gb");
    }
    /******/

    FileManager fm("deep", "GIRL");
    fm.createDir();
    //    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);

    /*** Set up MDP ***/
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

    BasisFunctions basis;
    for (int i = 0; i < actions.size() -1; ++i)
    {
        basis.push_back(new AndConditionBasisFunction(pf0,2,i));
        basis.push_back(new AndConditionBasisFunction(pfs1,2,i));
        basis.push_back(new AndConditionBasisFunction(pfs2,2,i));
        basis.push_back(new AndConditionBasisFunction(pfs1s2,2,i));
        basis.push_back(new AndConditionBasisFunction(d2si,2,i));
        basis.push_back(new AndConditionBasisFunction(dsi,2,i));
    }

    DenseFeatures phi(basis);
    ParametricGibbsPolicy<DenseState> expertPolicy(actions, phi, 1);
    //---

    vec eReward(2);
    eReward(0) = 1;
    eReward(1) = 0;
#if 1
    /*** learn the optimal policy ***/
    int nbepperpol = 150;
    AdaptiveStep srule(0.01);
    WeightedSumRT rewardtr(eReward);
    REINFORCEAlgorithm<FiniteAction, DenseState> agent(expertPolicy, nbepperpol,
            srule, rewardtr);
    ReLe::Core<FiniteAction, DenseState> core(mdp, agent);
    core.getSettings().loggerStrategy = new WriteStrategy<FiniteAction, DenseState>(
        fm.addPath("gradient_log_learning.log"),
        WriteStrategy<FiniteAction, DenseState>::AGENT,
        true /*delete file*/
    );

    int horiz = mdp.getSettings().horizon;
    core.getSettings().episodeLenght = horiz;

    int nbUpdates = 1550;
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
                cout << "mean score: " << J.t() << endl;
                cout << "MORL score: " << J.t() * eReward << endl;
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


    PolicyEvalAgent<FiniteAction, DenseState> expert(expertPolicy);

    /* Generate LQR expert dataset */
    Core<FiniteAction, DenseState> expertCore(mdp, expert);
    CollectorStrategy<FiniteAction, DenseState> collection;
    expertCore.getSettings().loggerStrategy = &collection;
    expertCore.getSettings().episodeLenght = 50;
    expertCore.getSettings().testEpisodeN = 100;
    expertCore.runTestEpisodes();


    /* Learn weight with GIRL */
    Deep_IRL_Reward rewardRegressor;
    Dataset<FiniteAction,DenseState>& data = collection.data;
    GIRL<FiniteAction,DenseState> irlAlg(data, expertPolicy, rewardRegressor,
                                         mdp.getSettings().gamma, atype);

    //Run MWAL
    irlAlg.run();
    arma::vec w = irlAlg.getWeights();

    cout << "Computed weights: " << endl << w.t() << endl;

    char filename[100];
    sprintf(filename, "girl_%s.log", alg);
    ofstream outf(fm.addPath(filename), ios_base::app);
    outf << w.t();


    vec grad, grad2, grad3;
    mat dgrad3;
    GradientFromDataWorker<FiniteAction,DenseState> gdw(data, expertPolicy, rewardRegressor, mdp.getSettings().gamma);
    if (atype == IRLGradType::R)
    {
        cout << "PG REINFORCE" << endl;
        grad3 = irlAlg.ReinforceGradient(dgrad3);
        rewardRegressor.setParameters(eReward);
        grad = gdw.ReinforceGradient();
        rewardRegressor.setParameters(w);
        grad2 = gdw.ReinforceGradient();
    }
    else if (atype == IRLGradType::RB)
    {
        cout << "PG REINFORCE BASE" << endl;
        grad3 = irlAlg.ReinforceBaseGradient(dgrad3);
        rewardRegressor.setParameters(eReward);
        grad = gdw.ReinforceBaseGradient();
        rewardRegressor.setParameters(w);
        grad2 = gdw.ReinforceBaseGradient();
    }
    else if (atype == IRLGradType::G)
    {
        cout << "PG GPOMDP" << endl;
        grad3 = irlAlg.GpomdpGradient(dgrad3);
        rewardRegressor.setParameters(eReward);
        grad = gdw.GpomdpGradient();
        rewardRegressor.setParameters(w);
        grad2 = gdw.GpomdpGradient();
    }
    else if (atype == IRLGradType::GB)
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

    vec v = linspace<vec>(0, 1, 30);
    ofstream ooo(fm.addPath("objective.log"));
    for (int i = 0; i < v.n_elem; ++i)
//        for (int j = 0; j < v.n_elem; ++j)
    {
        vec x(2);
        x(0) = v[i];
        x(1) = 1 - v[i];
        if (atype == IRLGradType::R)
        {
            rewardRegressor.setParameters(x);
            grad3 = irlAlg.ReinforceGradient(dgrad3);
        }
        else if (atype == IRLGradType::RB)
        {
            rewardRegressor.setParameters(x);
            grad3 = irlAlg.ReinforceBaseGradient(dgrad3);
        }
        else if (atype == IRLGradType::G)
        {
            rewardRegressor.setParameters(x);
            grad3 = irlAlg.GpomdpGradient(dgrad3);
        }
        else if (atype == IRLGradType::GB)
        {
            rewardRegressor.setParameters(x);
            grad3 = irlAlg.GpomdpBaseGradient(dgrad3);
        }
        double val = norm(grad3,2)*norm(grad3,2)*0.5;
        ooo << x(0) << "\t" << x(1) << "\t" << val << endl;
    }
    ooo.close();

    std::vector<IRLParametricReward<FiniteAction,DenseState>*> rewards;
    DEEP_R1 r1;
    DEEP_R2 r2;
    rewards.push_back(&r1);
    rewards.push_back(&r2);

    PlaneGIRL<FiniteAction,DenseState> pgirl(data, expertPolicy, rewards,
            mdp.getSettings().gamma, atype);

    pgirl.run();

    cout << pgirl.getWeights().t();

    return 0;
}
