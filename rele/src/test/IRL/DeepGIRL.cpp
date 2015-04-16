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

#include "DeepSeaTreasure.h"
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
        return weights(0)*deep_reward_treasure(s) - weights(1);
    }

    double operator()(const Reward& r)
    {
        return weights(0)*r[0] - weights(1)*r[1];
    }

    arma::mat diff(DenseState& s, FiniteAction& a, DenseState& ns)
    {
        arma::mat m(1,2);
        m(0) = deep_reward_treasure(s);
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


int main(int argc, char *argv[])
{
    //    RandomGenerator::seed(12354);

    FileManager fm("lqr", "GIRL");
    fm.createDir();
    //    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);

    /*** Set up MDP ***/
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
    }

    LinearApproximator regressor(mdp.getSettings().continuosStateDim + 1, basis);
    ParametricGibbsPolicy<DenseState> expertPolicy(actions, &regressor, 1);
    //---

#if 1
    /*** learn the optimal policy ***/
    int nbepperpol = 150;
    AdaptiveStep srule(0.001);
    vec eReward(2);
    eReward(0) = 0.6;
    eReward(1) = 0.4;
    WeightedSumRT rewardtr(eReward);
    GPOMDPAlgorithm<FiniteAction, DenseState> agent(expertPolicy, nbepperpol,
            mdp.getSettings().horizon, srule, rewardtr);
    ReLe::Core<FiniteAction, DenseState> core(mdp, agent);
    core.getSettings().loggerStrategy = new WriteStrategy<FiniteAction, DenseState>(
        fm.addPath("gradient_log_learning.log"),
        WriteStrategy<FiniteAction, DenseState>::AGENT,
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
    param << -0.0783 << -0.1144 << -0.2249 << -0.3522 <<  0.0056 << -0.0499 <<  0.8050 <<  1.1729 <<  0.0149 << -0.1588 <<  0.4539 <<  0.5678 <<  0.0112 <<  0.0357 << -0.0725 << -0.1477 <<  0.0047 << -0.0005;
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
    GIRL<FiniteAction,DenseState> irlAlg(data, expertPolicy, rewardRegressor, mdp.getSettings().gamma,
                                         GIRL<FiniteAction,DenseState>::AlgType::GB);

    //Run MWAL
    irlAlg.run();
    arma::vec w = irlAlg.getWeights();

    cout << "Computed weights: " << endl << w.t() << endl;

    ofstream outf(fm.addPath("girl.log"), ios_base::app);
    outf << w.t();

    IndexRT rt(0);
    GradientFromDataWorker<FiniteAction,DenseState> gdw(data, expertPolicy, rt, mdp.getSettings().gamma);
    arma::vec grad = gdw.GpomdpBaseGradient();

    rewardRegressor.setParameters(w);
    GradientFromDataWorker<FiniteAction,DenseState> gdw2(data, expertPolicy, rewardRegressor, mdp.getSettings().gamma);
    arma::vec grad2 = gdw2.GpomdpBaseGradient();

    cout << "Gradient Original: " << endl << grad.t() << endl;
    cout << "Gradient weights IRL: " << endl << grad2.t() << endl;

    cout << norm(grad2,2)*norm(grad2,2)*0.5 << endl;

    return 0;
}
