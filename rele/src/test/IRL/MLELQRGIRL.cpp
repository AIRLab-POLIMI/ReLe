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
#include "basis/PolynomialFunction.h"

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

#include "MLE.h"

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


void help()
{
    cout << "lqr_GMGIRL [algorithm]" << endl;
    cout << " - algorithm: r, rb, g, gb (default)" << endl;
}

int main(int argc, char *argv[])
{
    //    RandomGenerator::seed(45423424);
    RandomGenerator::seed(8763575);

    /*** check inputs ***/

    /******/
    IRLGradType atype = IRLGradType::RB;
    vec eReward(2);
    eReward(0) = 0.65;
    eReward(1) = 0.35;
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
//    vec initialState(1);
//    initialState[0] = 2;
//    mdp.setInitialState(initialState);

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


    BasisFunctions basis = GaussianRbf::generate({5}, {-4,4});
//    BasisFunctions basis = PolynomialFunction::generate(3,1);
    for (int i = 0; i < basis.size(); ++i)
        std::cout << *(basis[i]) << endl;
    DenseFeatures phin(basis);
    arma::mat cov(1,1);
    cov(0,0) = 2;
    MVNPolicy policy(phin,cov);

//    MVNDiagonalPolicy policy(phi);


    MLE mle(policy, data);
    double vv[] = {0,0,0,0,0};
    arma::vec startVal(vv,5);
//    double vv[] = {0,0,0,0};
//    arma::vec startVal(vv,4);
    arma::vec pp = mle.solve(startVal);

    std::cerr << pp.t();
    policy.setParameters(pp);

    int count = 0;
    arma::mat F;
    for (int ep = 0; ep < nbEpisodes; ++ep) //TO AVOID OVERFITTING
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

//    //compute importance weights
//    arma::vec IWOrig;
//    for (int ep = 0; ep < nbEpisodes; ++ ep)
//    {
//        int nbSteps = data[ep].size();
//        for (int t = 0; t < nbSteps; ++t)
//        {
//            Transition<DenseAction, DenseState>& tr = data[ep][t];
//            arma::vec iw(1);
//            iw(0) = policy(tr.x,tr.u)/tmpPolicy(tr.x,tr.u);
//            IWOrig = arma::join_vert(IWOrig, iw);
//        }
//    }
//    IWOrig.save(fm.addPath("iworig.log"), arma::raw_ascii);


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
