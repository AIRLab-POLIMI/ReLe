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

void help()
{
    cout << "lqr_GIRL [algorithm]" << endl;
    cout << " - algorithm: r, rb, g, gb (default)" << endl;
}

int main(int argc, char *argv[])
{
    RandomGenerator::seed(545404224);

    /*** check inputs ***/
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
    }
    else
    {
        atype = GIRL<DenseAction,DenseState>::AlgType::GB;
    }
    /******/

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


    vec eReward(2);
    eReward(0) = Q(0,0);
    eReward(1) = R(0,0);

    PolynomialFunction* pf = new PolynomialFunction(1,1);
    cout << *pf << endl;
    DenseFeatures phi(pf);
//    DetLinearPolicy<DenseState> expertPolicy(phi);
    NormalPolicy expertPolicy(1, phi);

#if 0
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
#endif

    vec param(1);
    param[0] = -0.390388203202208; //0.2, 0.8
    expertPolicy.setParameters(param);

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
    GIRL<DenseAction,DenseState> irlAlg(data, expertPolicy, rewardRegressor,
                                        mdp.getSettings().gamma, atype);

    //Run MWAL
    irlAlg.run();
    arma::vec w = irlAlg.getWeights();

    cout << "Computed weights: " << endl << w.t() << endl;

    ofstream outf(fm.addPath("girl.log"), ios_base::app);
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

    vec v = linspace<vec>(0, 1, 30);
    ofstream ooo(fm.addPath("objective.log"));
    for (int i = 0; i < v.n_elem; ++i)
//        for (int j = 0; j < v.n_elem; ++j)
    {
        vec x(2);
        x(0) = v[i];
        x(1) = 1 - v[i];
        if (atype == GIRL<DenseAction,DenseState>::AlgType::R)
        {
            rewardRegressor.setParameters(x);
            grad3 = irlAlg.ReinforceGradient(dgrad3);
        }
        else if (atype == GIRL<DenseAction,DenseState>::AlgType::RB)
        {
            rewardRegressor.setParameters(x);
            grad3 = irlAlg.ReinforceBaseGradient(dgrad3);
        }
        else if (atype == GIRL<DenseAction,DenseState>::AlgType::G)
        {
            rewardRegressor.setParameters(x);
            grad3 = irlAlg.GpomdpGradient(dgrad3);
        }
        else if (atype == GIRL<DenseAction,DenseState>::AlgType::GB)
        {
            rewardRegressor.setParameters(x);
            grad3 = irlAlg.GpomdpBaseGradient(dgrad3);
        }
        double val = norm(grad3,2)*norm(grad3,2)*0.5;
        ooo << x(0) << "\t" << x(1) << "\t" << val << endl;
    }
    ooo.close();

    return 0;
}
