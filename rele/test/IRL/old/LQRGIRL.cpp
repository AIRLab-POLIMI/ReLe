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
#include "features/DenseFeatures.h"
#include "DifferentiableNormals.h"
#include "basis/IdentityBasis.h"
#include "features/SparseFeatures.h"

#include "LQR.h"
#include "LQRsolver.h"
#include "PolicyEvalAgent.h"
#include "algorithms/GIRL.h"
#include "algorithms/PGIRL.h"

#include "ParametricRewardMDP.h"
#include "RandomGenerator.h"
#include "FileManager.h"

#include "policy_search/gradient/onpolicy/FunctionGradient.h"
#include "policy_search/gradient/PolicyGradientAlgorithm.h"

#include <boost/timer/timer.hpp>

using namespace boost::timer;
using namespace std;
using namespace ReLe;
using namespace arma;

class LQR_1D_WS : public IRLParametricReward<DenseAction, DenseState>,
    public RewardTransformation
{
public:

    LQR_1D_WS()
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

class LQR_1D_R1 : public IRLParametricReward<DenseAction, DenseState>
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
class LQR_1D_R2 : public IRLParametricReward<DenseAction, DenseState>
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

class LQR_ND_WS : public IRLParametricReward<DenseAction, DenseState>
{
public:

    LQR_ND_WS(LQR& mdp)
        : lqr(mdp)
    {
        weights.set_size(lqr.getSettings().rewardDim);
    }

    double operator()(DenseState& s, DenseAction& a, DenseState& ns)
    {
        int dim = lqr.Q.size();
        double val = 0.0;
        arma::vec& x = s;
        arma::vec& u = a;
        for (int i = 0; i < dim; ++i)
        {
            arma::mat& R = lqr.R[i];
            arma::mat& Q = lqr.Q[i];
            arma::mat J = (x.t() * Q * x + u.t() * R * u) * weights(i);
            val -= J(0,0);
        }
        return val;
    }

    arma::mat diff(DenseState& s, DenseAction& a, DenseState& ns)
    {
        int dim = lqr.Q.size();
        arma::vec& x = s;
        arma::vec& u = a;
        arma::mat m(1,dim);
        for (int i = 0; i < dim; ++i)
        {
            arma::mat& R = lqr.R[i];
            arma::mat& Q = lqr.Q[i];
            arma::mat J = -(x.t() * Q * x + u.t() * R * u);
            m(0,i) = J(0,0);
        }
        return m;
    }

private:
    LQR& lqr;
};

class LQR_ND_R : public IRLParametricReward<DenseAction, DenseState>
{
public:
    LQR_ND_R(LQR& mdp, unsigned int idx)
        : lqr(mdp), idx(idx)
    {
    }

    double operator()(DenseState& s, DenseAction& a, DenseState& ns)
    {
        arma::vec& x = s;
        arma::vec& u = a;
        arma::mat& R = lqr.R[idx];
        arma::mat& Q = lqr.Q[idx];
        arma::mat J = -(x.t() * Q * x + u.t() * R * u);
        return J(0,0);
    }

    arma::mat diff(DenseState& s, DenseAction& a, DenseState& ns)
    {
        return arma::mat();
    }
private:
    unsigned int idx;
    LQR& lqr;
};


void help()
{
    cout << "lqr_GIRL [algorithm]" << endl;
    cout << " - algorithm: r, rb, g, gb (default)" << endl;
}

int main(int argc, char *argv[])
{
//    RandomGenerator::seed(45423424);
//    RandomGenerator::seed(8763575);

    /*** check inputs ***/
    IRLGradType atype;
    vec eReward;
    int nbEpisodes;
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
        else if (strcmp(argv[1], "enac") == 0)
        {
            cout << "GIRL ENAC" << endl;
            atype = IRLGradType::ENAC;
        }
        else if (strcmp(argv[1], "natr") == 0)
        {
            cout << "GIRL NATR" << endl;
            atype = IRLGradType::NATR;
        }
        else if (strcmp(argv[1], "natrb") == 0)
        {
            cout << "GIRL NATR BASE" << endl;
            atype = IRLGradType::NATRB;
        }
        else if (strcmp(argv[1], "natg") == 0)
        {
            cout << "GIRL NATG" << endl;
            atype = IRLGradType::NATG;
        }
        else if (strcmp(argv[1], "natgb") == 0)
        {
            cout << "GIRL NATG BASE" << endl;
            atype = IRLGradType::NATGB;
        }
        else
        {
            std::cout << "Error unknown argument " << argv[1] << std::endl;
            help();
            exit(1);
        }
        strcpy(gtypestr, argv[1]);

        nbEpisodes = atoi(argv[2]);
        cout << "Episodes: " << nbEpisodes << endl;
        int nw = atoi(argv[3]);
        eReward.set_size(nw);
        for (int i = 0; i < nw; ++i)
            eReward(i) = atof(argv[4+i]);
    }
    else
    {
        atype = IRLGradType::GB;
    }
    /******/

    FileManager fm("lqr", "TESTGIRL");
    fm.createDir();
    //    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);

    /* Learn lqr correct policy */
#if 0

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

    IdentityBasis* pf = new IdentityBasis(0);
    DenseFeatures phi(pf);
    NormalPolicy expertPolicy(1, phi);

    vec eReward(2);
    eReward(0) = Q(0,0);
    eReward(1) = R(0,0);
#else

    int dim = eReward.n_elem;
    LQR mdp(dim, dim);

    BasisFunctions basis;
    for (int i = 0; i < dim; ++i)
    {
        basis.push_back(new IdentityBasis(i));
    }

    SparseFeatures phi;
    phi.setDiagonal(basis);

    MVNPolicy expertPolicy(phi);

    //    vec eReward(dim);
    //    eReward(0) = 0.05;
    //    eReward(1) = 0.7;
    //    eReward(2) = 0.25;
#endif



    /*** solve the problem in exact way ***/
    LQRsolver solver(mdp,phi);
    solver.setRewardWeights(eReward);
//    solver.solve();
    mat K = solver.computeOptSolution();
    arma::vec p = K.diag();
    expertPolicy.setParameters(p);

    std::cout << "Rewards: ";
    for (int i = 0; i < eReward.n_elem; ++i)
    {
        std::cout << eReward(i) << " ";
    }
    std::cout << "| Params: " << expertPolicy.getParameters().t() << std::endl;


    PolicyEvalAgent<DenseAction, DenseState> expert(expertPolicy);

    /* Generate LQR expert dataset */
    Core<DenseAction, DenseState> expertCore(mdp, expert);
    CollectorStrategy<DenseAction, DenseState> collection;
    expertCore.getSettings().loggerStrategy = &collection;
    expertCore.getSettings().episodeLength = mdp.getSettings().horizon;
    expertCore.getSettings().testEpisodeN = nbEpisodes;
    expertCore.runTestEpisodes();


    //save data
    Dataset<DenseAction,DenseState>& data = collection.data;
    ofstream datafile(fm.addPath("data.log"), ios_base::out);
    datafile << std::setprecision(OS_PRECISION);
    data.writeToStream(datafile);
    datafile.close();


    /* Learn weight with GIRL */
#if 0
    LQR_IRL_Reward rewardRegressor;
#else
    LQR_ND_WS rewardRegressor(mdp);
#endif
    GIRL<DenseAction,DenseState> irlAlg(data, expertPolicy, rewardRegressor,
                                        mdp.getSettings().gamma, atype);

    ofstream timefile(fm.addPath("timer.log"));


    //Run MWAL
    cpu_timer timer;
    timer.start();
    irlAlg.run();
    timer.stop();
    arma::vec gnormw = irlAlg.getWeights();

    timefile << timer.format(10, "%w") << std::endl;

    cout << "Weights (gnorm): " << gnormw.t();

    char name[100];
    sprintf(name, "girl_gnorm_%s.log", gtypestr);
    ofstream outf(fm.addPath(name), std::ofstream::out | std::ofstream::app);
    outf << std::setprecision(OS_PRECISION);
    for (int i = 0; i < gnormw.n_elem; ++i)
    {
        outf << gnormw[i] << " ";
    }
    outf.close();

    // TEST GRADIENTS
    vec grad, grad2, grad3;
    mat dgrad3;
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


//    //    GENERATES GRAPH
    vec v = linspace<vec>(0, 1, 30);
    char objname[100];
    sprintf(objname, "objective_%s.log", gtypestr);
    ofstream ooo(fm.addPath(objname));
    for (int i = 0; i < v.n_elem; ++i)
//        for (int j = 0; j < v.n_elem; ++j)
    {
        vec x(2);
        x(0) = v[i];
        x(1) = 1 - v[i];
//            x(1) = v[j];
//            x(2) = 1 - v[i] - v[j];
//            if (x(2) >= 0)
//            {
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
        else if (atype == IRLGradType::ENAC)
        {
            rewardRegressor.setParameters(x);
            grad3 = irlAlg.ENACGradient(dgrad3);
        }
        else if (atype == IRLGradType::NATR)
        {
            rewardRegressor.setParameters(x);
            grad3 = irlAlg.NaturalGradient(dgrad3);
        }
        else if (atype == IRLGradType::NATRB)
        {
            rewardRegressor.setParameters(x);
            grad3 = irlAlg.NaturalGradient(dgrad3);
        }
        else if (atype == IRLGradType::NATG)
        {
            rewardRegressor.setParameters(x);
            grad3 = irlAlg.NaturalGradient(dgrad3);
        }
        else if (atype == IRLGradType::NATGB)
        {
            rewardRegressor.setParameters(x);
            grad3 = irlAlg.NaturalGradient(dgrad3);
        }
        //                else if (atype == IRLGradType::ENACB)
        //                {
        //                    rewardRegressor.setParameters(x);
        //                    grad3 = irlAlg.GpomdpBaseGradient(dgrad3);
        //                }
        double val = norm(grad3,2)*norm(grad3,2)*0.5;
//                ooo << x(0) << "\t" << x(1) << "\t" << val << endl;
        ooo << x(0) << "\t" << val << endl;
//            }
    }
    ooo.close();

    std::vector<IRLParametricReward<DenseAction,DenseState>*> rewards;
#if 0
    LQR_R1 r1;
    LQR_R2 r2;
    rewards.push_back(&r1);
    rewards.push_back(&r2);
#else
    for (int i = 0; i < dim; ++i)
    {
        rewards.push_back(new LQR_ND_R(mdp, i));
    }
#endif

    PlaneGIRL<DenseAction,DenseState> pgirl(data, expertPolicy, rewards,
                                            mdp.getSettings().gamma, atype);

    timer.start();
    pgirl.run();
    timer.stop();
    timefile << timer.format(10, "%w") << std::endl;

    timefile.close();

    cout << "Weights (plane): " << pgirl.getWeights().t();

    sprintf(name, "girl_plane_%s.log", gtypestr);
    outf.open(fm.addPath(name), std::ofstream::out | std::ofstream::app);
    outf << std::setprecision(OS_PRECISION);
    arma::vec planew = pgirl.getWeights();
    for (int i = 0; i < planew.n_elem; ++i)
    {
        outf << planew[i] << " ";
    }
    outf.close();

    return 0;
}
