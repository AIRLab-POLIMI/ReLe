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
#include "PolicyEvalAgent.h"

#include "parametric/differentiable/GibbsPolicy.h"
#include "features/DenseFeatures.h"
#include "features/SparseFeatures.h"
#include "DifferentiableNormals.h"
#include "basis/IdentityBasis.h"
#include "basis/GaussianRbf.h"
#include "basis/PolynomialFunction.h"

#include "ParametricRewardMDP.h"
#include "RandomGenerator.h"
#include "FileManager.h"

#include "MountainCar.h"
#include "MLE.h"

#include <boost/timer/timer.hpp>

using namespace boost::timer;
using namespace std;
using namespace ReLe;
using namespace arma;

class mountain_car_manual_policy : public Policy<FiniteAction, DenseState>
{
public:
    unsigned int operator()(const arma::vec& state)
    {
        double speed = state(MountainCar::StateLabel::velocity);
        if (speed <= 0)
            return 0;
        else
            return 2;
    }

    double operator()(const arma::vec& state, const unsigned int& action)
    {
        double speed = state(MountainCar::StateLabel::velocity);
        if (speed <= 0 && action == 0)
            return 1;
        else if(speed > 0 && action == 2)
            return 1;
        else
            return 0;
    }

    // Policy interface
public:
    string getPolicyName()
    {
        return "mountain_car_manual_policy";
    }
    string getPolicyHyperparameters()
    {
        return "";
    }
    string printPolicy()
    {
        return "";
    }

    mountain_car_manual_policy* clone()
    {
        return new mountain_car_manual_policy();
    }
};

int main(int argc, char *argv[])
{
    FileManager fm("mc", "GIRL");
    fm.createDir();
    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);

    MountainCar mdp;

    /*** define expert's policy ***/
    mountain_car_manual_policy expertPolicy;

    /*** get expert's trajectories ***/
    PolicyEvalAgent<FiniteAction, DenseState> expert(expertPolicy);
    Core<FiniteAction, DenseState> expertCore(mdp, expert);
    CollectorStrategy<FiniteAction, DenseState> collection;
    expertCore.getSettings().loggerStrategy = &collection;
    expertCore.getSettings().episodeLenght = mdp.getSettings().horizon;
    expertCore.getSettings().testEpisodeN = 100;
    expertCore.runTestEpisodes();


    /*** save data ***/
    Dataset<FiniteAction,DenseState>& data = collection.data;
    ofstream datafile(fm.addPath("data.log"), ios_base::out);
    datafile << std::setprecision(OS_PRECISION);
    data.writeToStream(datafile);
    datafile.close();

    /*** define policy for MLE ***/
    vector<FiniteAction> actions;
    for (int i = 0; i < mdp.getSettings().finiteActionDim; ++i)
        actions.push_back(FiniteAction(i));
    BasisFunctions basis = PolynomialFunction::generate(1,mdp.getSettings().continuosStateDim);
    SparseFeatures phi(basis, actions.size());
    ParametricGibbsPolicy<DenseState> mlePolicy(actions, phi, 1);

    /*** compute MLE ***/
    MLE<FiniteAction,DenseState> mle(mlePolicy, data);
    arma::vec startVal(mlePolicy.getParametersSize(),arma::fill::ones);
    arma::vec pp = mle.solve(startVal);

    std::cerr << pp.t();
    mlePolicy.setParameters(pp);


    return 0;
}
