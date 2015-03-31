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
#include "BasisFunctions.h"
#include "DifferentiableNormals.h"
#include "basis/PolynomialFunction.h"

#include "LQR.h"
#include "PolicyEvalAgent.h"
#include "algorithms/MWAL.h"
#include "policy_search/PGPE/PGPE.h"
#include "ParametricRewardMDP.h"

using namespace std;
using namespace ReLe;
using namespace arma;


class MWALBasis : public BasisFunction
{

public:
	MWALBasis(unsigned int index, double max, double min) :
		index(index), max(max), min(min)
	{

	}

	virtual double operator()(const arma::vec& input)
	{
		double value = input[index];
		return abs(value - min) / (max - min);
	}

	virtual void writeOnStream(std::ostream& out)
	{

	}

	virtual void readFromStream(std::istream& in)
	{

	}

private:
	unsigned int index;
	double max;
	double min;
};

int main(int argc, char *argv[])
{
    /* Learn lqr correct policy */
    LQR mdp(1,1); //with these settings the optimal value is -0.6180 (for the linear policy)

    PolynomialFunction* pf = new PolynomialFunction(1,1);
    DenseBasisVector basis;
    basis.push_back(pf);
    LinearApproximator regressor(mdp.getSettings().continuosStateDim, basis);
    DetLinearPolicy<DenseState> policy(&regressor);
    vec param(1);
    param[0] = -0.6180;
    policy.setParameters(param);
    PolicyEvalAgent<DenseAction, DenseState, DetLinearPolicy<DenseState>> expert(policy);

    /* Generate LQR expert dataset */
    Core<DenseAction, DenseState> expertCore(mdp, expert);
    CollectorStrategy<DenseAction, DenseState> collection;
    expertCore.getSettings().loggerStrategy = &collection;
    expertCore.getSettings().episodeLenght = 50;
    expertCore.getSettings().testEpisodeN = 1000;
    expertCore.runTestEpisodes();


    /* Learn weight with MWAL */

    //Compute expert feature expectations
    arma::vec muE = collection.data.computefeatureExpectation(basis, mdp.getSettings().gamma);

    // Create a parametric MDP
    DenseBasisVector rewardBasis;
    MWALBasis* bf = new MWALBasis(-1, +20, -20);
    rewardBasis.push_back(bf);
    LinearApproximator rewardRegressor(1, rewardBasis);
    ParametricRewardMDP<DenseAction, DenseState> prMdp(mdp, rewardRegressor);

    //Create an agent to solve the mdp direct problem
    int nparams = basis.size();
    arma::vec mean(nparams, fill::zeros);
    mean[0] = -0.1;

    int nbepperpol = 1, nbpolperupd = 100;
    arma::mat cov(nparams, nparams, arma::fill::eye);
    cov *= 0.1;
    ParametricNormal dist(mean, cov);
    PGPE<DenseAction, DenseState> agent(dist, policy, nbepperpol, nbpolperupd, 0.002, false);

    //Setup the solver
    int nbUpdates = 600;
    int episodes  = nbUpdates*nbepperpol*nbpolperupd;

    Core<DenseAction, DenseState> core(prMdp, agent);

    //Run MWAL
    unsigned int T = 100;
    double gamma = prMdp.getSettings().gamma;

    MWAL<DenseAction, DenseState> irlAlg(rewardBasis, rewardRegressor, core, T, gamma, muE);
    irlAlg.run();
    arma::vec w = irlAlg.getWeights();
    cout << w << endl;


    return 0;
}
