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

using namespace std;
using namespace ReLe;
using namespace arma;

class LQR_IRL_Reward : public IRLParametricReward<DenseAction, DenseState>
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
    LQR mdp(1,1); //with these settings the optimal value is -0.6180 (for the linear policy)
//    vec initialState(1);
//    initialState[0] = -5;
//    mdp.setInitialState(initialState);

    PolynomialFunction* pf = new PolynomialFunction(1,1);
    DenseBasisVector basis;
    basis.push_back(pf);
    LinearApproximator expertRegressor(mdp.getSettings().continuosStateDim, basis);
//    DetLinearPolicy<DenseState> expertPolicy(&expertRegressor);
    NormalPolicy expertPolicy(1,&expertRegressor);
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
    GIRL<DenseAction,DenseState> irlAlg(data, expertPolicy, rewardRegressor, mdp.getSettings().gamma,
                                        GIRL<DenseAction,DenseState>::AlgType::GB);

    //Run MWAL
    irlAlg.run();
    arma::vec w = irlAlg.getWeights();

    cout << "Computed weights: " << endl << w.t() << endl;

    ofstream outf(fm.addPath("girl.log"), ios_base::app);
    outf << w.t();

    return 0;
}
