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
#include "parametric/differentiable/LinearPolicy.h"
#include "parametric/differentiable/NormalPolicy.h"
#include "BasisFunctions.h"
#include "DifferentiableNormals.h"
#include "basis/PolynomialFunction.h"

#include "LQR.h"
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

class LQR_IRL_Reward : public IRLParametricReward<DenseAction, DenseState>,
    public RewardTransformation
{
public:

    LQR_IRL_Reward()
    {
        weights.set_size(2);
    }

    double operator()(DenseState& s, DenseAction& a, DenseState& ns)
    {
        return -(weights(0)*s(0)*s(0)+weights(1)*a(0)*a(0));
    }

    double operator()(const Reward& r)
    {
        return r[0];
    }

    arma::mat diff(DenseState& s, DenseAction& a, DenseState& ns)
    {
        arma::mat m(1,2);
        m(0) = -s(0)*s(0);
        m(1) = -a(0)*a(0);
        return m;
    }
};

int main(int argc, char *argv[])
{
//    RandomGenerator::seed(12354);

    FileManager fm("lqr", "GIRL");
    fm.createDir();
//    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);

    /* Learn lqr correct policy */
    arma::mat A(1,1), B(1,1), Q(1,1), R(1,1);
    A(0,0) = 1;
    B(0,0) = 1;
    Q(0,0) = 0.2;
    R(0,0) = 0.8;
    std::vector<arma::mat> Qv(1, Q);
    std::vector<arma::mat> Rv(1, R);
    LQR mdp(A,B,Qv,Rv);
//    vec initialState(1);
//    initialState[0] = -5;
//    mdp.setInitialState(initialState);

    PolynomialFunction* pf = new PolynomialFunction(1,1);
    cout << *pf << endl;
    DenseBasisVector basis;
    basis.push_back(pf);
    LinearApproximator expertRegressor(mdp.getSettings().continuosStateDim, basis);
//    DetLinearPolicy<DenseState> expertPolicy(&expertRegressor);
    NormalPolicy expertPolicy(1,&expertRegressor);

    //learn the optimal policy
    int nbepperpol = 100;
    AdaptiveStep srule(0.0001);
    GPOMDPAlgorithm<DenseAction, DenseState> agent(expertPolicy, nbepperpol,
            mdp.getSettings().horizon, srule, 0);
    ReLe::Core<DenseAction, DenseState> core(mdp, agent);
    core.getSettings().loggerStrategy = new WriteStrategy<DenseAction, DenseState>(
        fm.addPath("gradient_log_learning.log"),
        WriteStrategy<DenseAction, DenseState>::AGENT,
        true /*delete file*/
    );

    int horiz = mdp.getSettings().horizon;
    core.getSettings().episodeLenght = horiz;

    int nbUpdates = 60;
    int episodes  = nbUpdates*nbepperpol;
    double every, bevery;
    every = bevery = 0.1; //%
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
    cout << expertPolicy.getParameters().t();

//    vec param(1);
//    param[0] = -0.390388203202208; //0.2, 0.8
//    expertPolicy.setParameters(param);

    PolicyEvalAgent<DenseAction, DenseState> expert(expertPolicy);

    /* Generate LQR expert dataset */
    Core<DenseAction, DenseState> expertCore(mdp, expert);
    CollectorStrategy<DenseAction, DenseState> collection;
    expertCore.getSettings().loggerStrategy = &collection;
    expertCore.getSettings().episodeLenght = 50;
    expertCore.getSettings().testEpisodeN = 100;
    expertCore.runTestEpisodes();


    /* Learn weight with GIRL */
    LQR_IRL_Reward rewardRegressor;
    Dataset<DenseAction,DenseState>& data = collection.data;
    GIRL<DenseAction,DenseState> irlAlg(data, expertPolicy, rewardRegressor, mdp.getSettings().gamma,
                                        GIRL<DenseAction,DenseState>::AlgType::GB);

    //Run MWAL
    irlAlg.run();
    arma::vec w = irlAlg.getWeights();

    cout << "Computed weights: " << endl << w.t() << endl;

    ofstream outf(fm.addPath("girl.log"), ios_base::app);
    outf << w.t();

    IndexRT rt(0);
    GradientFromDataWorker<DenseAction,DenseState> gdw(data, expertPolicy, rt, mdp.getSettings().gamma);
    arma::vec grad = gdw.GpomdpBaseGradient();

    rewardRegressor.setParameters(w);
    GradientFromDataWorker<DenseAction,DenseState> gdw2(data, expertPolicy, rewardRegressor, mdp.getSettings().gamma);
    arma::vec grad2 = gdw2.GpomdpBaseGradient();

    cout << "Gradient Original: " << endl << grad.t() << endl;
    cout << "Gradient weights IRL: " << endl << grad2.t() << endl;

    cout << norm(grad2,2)*norm(grad2,2)*0.5 << endl;

    return 0;
}
