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
#include "basis/IdentityBasis.h"

#include "LQR.h"
#include "LQRsolver.h"
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

class LQR_R1 : public IRLParametricReward<DenseAction, DenseState>
{
public:
    double operator()(DenseState& s, DenseAction& a, DenseState& ns)
    {
        return -s(0)*s(0);
    }

    arma::mat diff(DenseState& s, DenseAction& a, DenseState& ns)
    {
        return arma::mat();
    }
};
class LQR_R2 : public IRLParametricReward<DenseAction, DenseState>
{
public:
    double operator()(DenseState& s, DenseAction& a, DenseState& ns)
    {
        return -a(0)*a(0);
    }

    arma::mat diff(DenseState& s, DenseAction& a, DenseState& ns)
    {
        return arma::mat();
    }
};

void help()
{
    cout << "lqr_GIRL [algorithm]" << endl;
    cout << " - algorithm: r, rb, g, gb (default)" << endl;
}

int main(int argc, char *argv[])
{
//    RandomGenerator::seed(545404224);

    /*** check inputs ***/
    IRLGradType atype;
    char gtypestr[10];
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
        strcpy(gtypestr, argv[1]);
    }
    else
    {
        atype = IRLGradType::GB;
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

//    LQR mdp(2,2);


//    IdentityBasis* pf1 = new IdentityBasis(0);
//    IdentityBasis* pf2 = new IdentityBasis(1);
//    IdentityBasis* pf3 = new IdentityBasis(2);
//    BasisFunctions basis;
//    basis.push_back(pf1);
//    basis.push_back(pf2);
//    basis.push_back(pf3);

//    SparseFeatures phi;
//    phi.setDiagonal(basis);

//    arma::vec ss(3);
//    ss.randn();
//    std::cout << ss.t();
//    std::cout << phi(ss);


    vec eReward(2);
    eReward(0) = Q(0,0);
    eReward(1) = R(0,0);

    IdentityBasis* pf = new IdentityBasis(0);
    DenseFeatures phi(pf);
    NormalPolicy expertPolicy(1, phi);


    /*** solve the problem in exact way ***/
    LQRsolver solver(mdp,phi);
    solver.solve();
    DetLinearPolicy<DenseState>& pol = reinterpret_cast<DetLinearPolicy<DenseState>&>(solver.getPolicy());
    arma::vec p = pol.getParameters();
    expertPolicy.setParameters(p);

    std::cout << "Rewards: ";
    for (int i = 0; i < eReward.n_elem; ++i)
    {
        std::cout << eReward(i) << " ";
    }
    std::cout << "| Params: " << expertPolicy.getParameters().t() << std::endl;
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

    cout << "Weights (gnorm): " << w.t();

    char name[100];
    sprintf(name, "girl_gnorm_%s.log", gtypestr);
    ofstream outf(fm.addPath(name), std::ofstream::out | std::ofstream::app);
    outf << w.t();
    outf.close();

    // TEST GRADIENTS
//    vec grad, grad2, grad3;
//    mat dgrad3;
//    GradientFromDataWorker<DenseAction,DenseState> gdw(data, expertPolicy, rewardRegressor, mdp.getSettings().gamma);
//    if (atype == IRLGradType::R)
//    {
//        cout << "PG REINFORCE" << endl;
//        grad3 = irlAlg.ReinforceGradient(dgrad3);
//        rewardRegressor.setParameters(eReward);
//        grad = gdw.ReinforceGradient();
//        rewardRegressor.setParameters(w);
//        grad2 = gdw.ReinforceGradient();
//    }
//    else if (atype == IRLGradType::RB)
//    {
//        cout << "PG REINFORCE BASE" << endl;
//        grad3 = irlAlg.ReinforceBaseGradient(dgrad3);
//        rewardRegressor.setParameters(eReward);
//        grad = gdw.ReinforceBaseGradient();
//        rewardRegressor.setParameters(w);
//        grad2 = gdw.ReinforceBaseGradient();
//    }
//    else if (atype == IRLGradType::G)
//    {
//        cout << "PG GPOMDP" << endl;
//        grad3 = irlAlg.GpomdpGradient(dgrad3);
//        rewardRegressor.setParameters(eReward);
//        grad = gdw.GpomdpGradient();
//        rewardRegressor.setParameters(w);
//        grad2 = gdw.GpomdpGradient();
//    }
//    else if (atype == IRLGradType::GB)
//    {
//        cout << "PG GPOMDP BASE" << endl;
//        grad3 = irlAlg.GpomdpBaseGradient(dgrad3);
//        rewardRegressor.setParameters(eReward);
//        grad = gdw.GpomdpBaseGradient();
//        rewardRegressor.setParameters(w);
//        grad2 = gdw.GpomdpBaseGradient();
//    }
//    cout << "Gradient Original (" << eReward[0] << ", " << eReward[1] << "): " << endl << grad.t();
//    cout << "\tnorm2: " << norm(grad,2) << endl << endl;
//    cout << "Gradient weights IRL (" << w[0] << ", " << w[1] << "): " << endl << grad2.t();
//    cout << "\tnorm2: " << norm(grad2,2) << endl << endl;
//    cout << "Gradient computed by objective function: " << endl << grad3.t();
//    cout << "\tnorm2: " << norm(grad3,2) << endl;
//    cout << "Objective function (0.5*nomr(g,2)^2: " << norm(grad3,2)*norm(grad3,2)*0.5 << endl;


    //GENERATES GRAPH
//    vec v = linspace<vec>(0, 1, 30);
//    ofstream ooo(fm.addPath("objective.log"));
//    for (int i = 0; i < v.n_elem; ++i)
////        for (int j = 0; j < v.n_elem; ++j)
//    {
//        vec x(2);
//        x(0) = v[i];
//        x(1) = 1 - v[i];
//        if (atype == IRLGradType::R)
//        {
//            rewardRegressor.setParameters(x);
//            grad3 = irlAlg.ReinforceGradient(dgrad3);
//        }
//        else if (atype == IRLGradType::RB)
//        {
//            rewardRegressor.setParameters(x);
//            grad3 = irlAlg.ReinforceBaseGradient(dgrad3);
//        }
//        else if (atype == IRLGradType::G)
//        {
//            rewardRegressor.setParameters(x);
//            grad3 = irlAlg.GpomdpGradient(dgrad3);
//        }
//        else if (atype == IRLGradType::GB)
//        {
//            rewardRegressor.setParameters(x);
//            grad3 = irlAlg.GpomdpBaseGradient(dgrad3);
//        }
//        double val = norm(grad3,2)*norm(grad3,2)*0.5;
//        ooo << x(0) << "\t" << x(1) << "\t" << val << endl;
//    }
//    ooo.close();

    std::vector<IRLParametricReward<DenseAction,DenseState>*> rewards;
    LQR_R1 r1;
    LQR_R2 r2;
    rewards.push_back(&r1);
    rewards.push_back(&r2);

    PlaneGIRL<DenseAction,DenseState> pgirl(data, expertPolicy, rewards,
                                            mdp.getSettings().gamma, atype);

    pgirl.run();

    cout << "Weights (plane): " << pgirl.getWeights().t();

    sprintf(name, "girl_plane_%s.log", gtypestr);
    outf.open(fm.addPath(name), std::ofstream::out | std::ofstream::app);
    outf << pgirl.getWeights().t();
    outf.close();

    return 0;
}
