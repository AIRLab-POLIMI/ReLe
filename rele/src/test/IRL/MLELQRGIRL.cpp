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
#include "basis/GaussianRbf.h"

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

class MLE
{
public:
    MLE(ParametricPolicy<DenseAction,DenseState>& policy, Dataset<DenseAction,DenseState>& ds)
        : policy(policy), data(ds)
    {
    }

    arma::vec solve(arma::vec starting)
    {

        int dp = policy.getParametersSize();
        nlopt::opt optimizator;
        optimizator = nlopt::opt(nlopt::algorithm::LN_COBYLA, dp);
        optimizator.set_min_objective(MLE::wrapper, this);
        optimizator.set_xtol_rel(1e-6);
        optimizator.set_ftol_rel(1e-6);
        optimizator.set_maxeval(200);

        //optimize the function
        std::vector<double> parameters(dp, 0.0);
        for (int i = 0; i < dp; ++i)
            parameters[i] = starting[i];
        double minf;
        if (optimizator.optimize(parameters, minf) < 0)
        {
            printf("nlopt failed!\n");
            abort();
        }
        else
        {
            printf("found minimum = %0.10g\n", minf);

            arma::vec finalP(dp);
            for(int i = 0; i < dp; ++i)
            {
                finalP(i) = parameters[i];
            }

            return finalP;
        }
    }

    double objFunction(unsigned int n, const double* x, double* grad)
    {
        int dp = policy.getParametersSize();
        assert(dp == n);
        arma::vec params(x, dp);
        policy.setParameters(params);

        int nbEpisodes = data.size();
        double likelihood = 0.0;
        int counter = 0;
        for (int ep = 0; ep < nbEpisodes/2; ++ep) //TO AVOID OVERFITTING
        {
            int nbSteps = data[ep].size();
            for (int t = 0; t < nbSteps; ++t)
            {
                Transition<DenseAction, DenseState>& tr = data[ep][t];
                double prob = policy(tr.x,tr.u);
                prob = max(1e-10,prob);
                likelihood += log(prob);

                ++counter;
            }
        }
        likelihood /= counter;
        return -likelihood;
    }


    static double wrapper(unsigned int n, const double* x, double* grad,
                          void* o)
    {
        return reinterpret_cast<MLE*>(o)->objFunction(n, x, grad);
    }

private:
    ParametricPolicy<DenseAction,DenseState>& policy;
    Dataset<DenseAction,DenseState>& data;
};



void help()
{
    cout << "lqr_GMGIRL [algorithm]" << endl;
    cout << " - algorithm: r, rb, g, gb (default)" << endl;
}

int main(int argc, char *argv[])
{
    //    RandomGenerator::seed(45423424);
//    RandomGenerator::seed(8763575);

    /*** check inputs ***/

    /******/
    IRLGradType atype = IRLGradType::RB;
    vec eReward(2);
    eReward(0) = 0.4;
    eReward(1) = 0.6;
    char gtypestr[10];
    int nbEpisodes = 150;

    FileManager fm("lqr", "MLE");
    fm.createDir();
    //    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);

    /* Learn lqr correct policy */
    arma::mat A(1,1), B(1,1), Q(1,1), R(1,1);
    A(0,0) = 1;
    B(0,0) = 1;
    Q(0,0) = eReward(0);
    R(0,0) = eReward(1);
    std::vector<arma::mat> Qv(1, Q);
    std::vector<arma::mat> Rv(1, R);
    LQR mdp(A,B,Qv,Rv);
    vec initialState(1);
    initialState[0] = 2;
    mdp.setInitialState(initialState);

    IdentityBasis* pf = new IdentityBasis(0);
    DenseFeatures phi(pf);
    NormalPolicy tmpPolicy(1, phi);

    LQRsolver solver(mdp,phi,LQRsolver::Type::CLASSIC);
    solver.setRewardWeights(eReward);
    mat K = solver.computeOptSolution();
    arma::vec p = K.diag();
    std::cout << "optimal pol: " << p.t();
    tmpPolicy.setParameters(p);


    std::cout << "Rewards: ";
    for (int i = 0; i < eReward.n_elem; ++i)
    {
        std::cout << eReward(i) << " ";
    }
    std::cout << "| Params: " << tmpPolicy.getParameters().t() << std::endl;

    PolicyEvalAgent<DenseAction, DenseState> expert(tmpPolicy);

    /* Generate LQR expert dataset */
    Core<DenseAction, DenseState> expertCore(mdp, expert);
    CollectorStrategy<DenseAction, DenseState> collection;
    expertCore.getSettings().loggerStrategy = &collection;
    expertCore.getSettings().episodeLenght = mdp.getSettings().horizon;
    expertCore.getSettings().testEpisodeN = nbEpisodes;
    expertCore.runTestEpisodes();


    //save data
    Dataset<DenseAction,DenseState>& data = collection.data;
    ofstream datafile(fm.addPath("data.log"), ios_base::out);
    datafile << std::setprecision(OS_PRECISION);
    data.writeToStream(datafile);
    datafile.close();


    BasisFunctions basis = GaussianRbf::generate({5}, {-3,3});
    DenseFeatures phin(basis);
    arma::mat cov(1,1);
    cov(0,0) = 3;
    MVNPolicy policy(phin);

//    MVNDiagonalPolicy policy(phi);


    MLE mle(policy, data);
    double vv[] = {0,0,0,0,0,0};
    arma::vec startVal(vv,6);
    arma::vec pp = mle.solve(startVal);

    std::cerr << pp.t();
    policy.setParameters(pp);

    int count = 0;
    arma::mat F;
    for (int ep = 0; ep < nbEpisodes/2; ++ep) //TO AVOID OVERFITTING
    {
        int nbSteps = data[ep].size();
        for (int t = 0; t < nbSteps; ++t)
        {
            Transition<DenseAction, DenseState>& tr = data[ep][t];
            arma::vec aa = policy(tr.x);
            F = arma::join_horiz(F,aa);
            ++count;
        }
    }

    F.save(fm.addPath("datafit.log"), arma::raw_ascii);


    LQR_1D_WS rewardRegressor;
    GIRL<DenseAction,DenseState> irlAlg(data, policy, rewardRegressor,
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
    ofstream outf(fm.addPath(name), std::ofstream::out);
    outf << std::setprecision(OS_PRECISION);
    for (int i = 0; i < gnormw.n_elem; ++i)
    {
        outf << gnormw[i] << " ";
    }
    outf.close();

    return 0;
}
