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
#include "q_policy/e_Greedy.h"
#include "basis/IdentityBasis.h"
#include "basis/GaussianRbf.h"
#include "basis/PolynomialFunction.h"
#include "basis/ConditionBasedFunction.h"

#include "ParametricRewardMDP.h"
#include "RandomGenerator.h"
#include "FileManager.h"

#include "MountainCar.h"
#include "MLE.h"
#include "batch/LSPI.h"
#include "algorithms/PGIRL.h"

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

class mc_reward : public IRLParametricReward<FiniteAction, DenseState>
{
public:
    mc_reward(arma::vec mu, double sigma_position, double sigma_velocity, unsigned int actionIdx)
        : mu(mu), sigma_position(sigma_position), sigma_velocity(sigma_velocity), actionIdx(actionIdx)
    {
    }


    double operator()(DenseState& s, FiniteAction& a, DenseState& ns)
    {
        if (a.getActionN() != actionIdx)
            return 0;
        int posIdx = MountainCar::StateLabel::position;
        int velIdx = MountainCar::StateLabel::velocity;
        double A = - (s[posIdx] - mu[posIdx]) * (s[posIdx] - mu[posIdx]) / sigma_position;
        double B = - (s[velIdx] - mu[velIdx]) * (s[velIdx] - mu[velIdx]) / sigma_velocity;
        double val = exp(A + B);
        return val;
    }

    arma::mat diff(DenseState& s, FiniteAction& a, DenseState& ns)
    {
        return arma::mat();
    }

protected:
    arma::vec mu;
    double sigma_position, sigma_velocity;
    unsigned int actionIdx;
};

class mc_reward_one : public IRLParametricReward<FiniteAction, DenseState>
{
public:
    mc_reward_one(unsigned int actionIdx)
        : actionIdx(actionIdx)
    {
    }


    double operator()(DenseState& s, FiniteAction& a, DenseState& ns)
    {
        if (a.getActionN() != actionIdx)
            return 0;
        return 1.0;
    }

    arma::mat diff(DenseState& s, FiniteAction& a, DenseState& ns)
    {
        return arma::mat();
    }

protected:
    unsigned int actionIdx;
};

int main(int argc, char *argv[])
{
    FileManager fm("mc", "GIRL");
    fm.createDir();
    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);

    int nbEpisodes = 1;

    MountainCar mdp;

    /*** define expert's policy ***/
    mountain_car_manual_policy expertPolicy;

    /*** get expert's trajectories ***/
    PolicyEvalAgent<FiniteAction, DenseState> expert(expertPolicy);
    Core<FiniteAction, DenseState> expertCore(mdp, expert);
    CollectorStrategy<FiniteAction, DenseState> collection;
    expertCore.getSettings().loggerStrategy = &collection;
    expertCore.getSettings().episodeLenght = mdp.getSettings().horizon;
    expertCore.getSettings().testEpisodeN = nbEpisodes;
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
    //Replicate each basis for the actions (action-1 because the last is linearly dependent on the others)
    BasisFunctions bfs;
    for (int i = 0, ie = actions.size()-1; i < ie; ++i)
    {
        for (int k = 0, ke = basis.size(); k < ke; ++k)
        {
            bfs.push_back(new AndConditionBasisFunction(basis[k],2,i));
        }
    }
    //create basis vector
    DenseFeatures phi(bfs);
    //create policy
    ParametricGibbsPolicy<DenseState> mlePolicy(actions, phi, 1);
    //    arma::vec input = {3,2,2};
    //    cout << phi(input);


    /*** get only trailing info ***/
    int nbSampleMLE = 27;
    Dataset<FiniteAction,DenseState> dataExpert;
    Episode<FiniteAction,DenseState> episodeExpert;
    int budget = nbSampleMLE;
    for (int ep = 0; ep < data.size() && budget > 0; ++ep)
    {
        int nbSamples = data[ep].size();
        for (int t = nbSamples-1; t > nbSamples - 50 && t > 0 && budget > 0; --t)
        {
            episodeExpert.push_back(data[ep][t]);
            --budget;
        }
    }
    dataExpert.push_back(episodeExpert);

    /*** save MLE data ***/
    datafile.open(fm.addPath("mletraining.log"), ios_base::out);
    datafile << std::setprecision(OS_PRECISION);
    dataExpert.writeToStream(datafile);
    datafile.close();


    /*** compute MLE ***/
    MLE<FiniteAction,DenseState> mle(mlePolicy, dataExpert);
    arma::vec startVal(mlePolicy.getParametersSize(),arma::fill::ones);
    arma::vec pp = mle.solve(startVal);

    std::cerr << pp.t();
    mlePolicy.setParameters(pp);

#if EVAL_MLE
    /*** get mlePolicy trajectories ***/
    PolicyEvalAgent<FiniteAction, DenseState> mleEval(mlePolicy);
    Core<FiniteAction, DenseState> mleCore(mdp, mleEval);
    CollectorStrategy<FiniteAction, DenseState> collectionMle;
    mleCore.getSettings().loggerStrategy = &collectionMle;
    mleCore.getSettings().episodeLenght = mdp.getSettings().horizon;
    mleCore.getSettings().testEpisodeN = nbEpisodes;
    mleCore.runTestEpisodes();


    /*** save data ***/
    Dataset<FiniteAction,DenseState>& dataMle = collectionMle.data;
    datafile.open(fm.addPath("mledata.log"), ios_base::out);
    datafile << std::setprecision(OS_PRECISION);
    dataMle.writeToStream(datafile);
    datafile.close();
#endif

    /*** recover reward by IRL (PGIRL) ***/
    std::vector<IRLParametricReward<FiniteAction,DenseState>*> rewards;
    vec pos_linspace = linspace<vec>(-1.2,0.6,7);
    vec vel_linspace = linspace<vec>(-0.07,0.07,7);

    //-- meshgrid
    arma::mat xrow = vectorise(pos_linspace).t();
    arma::mat ycol = vectorise(vel_linspace);
    arma::mat xx_pos = repmat(xrow, ycol.n_rows, ycol.n_cols);
    arma::mat yy_vel = repmat(ycol, xrow.n_rows, xrow.n_cols);
    //--

    arma::vec pos_mesh = vectorise(xx_pos);
    arma::vec vel_mesh = vectorise(yy_vel);

    double sigma_position = 2*pow((0.6+1.2)/10.0,2);
    double sigma_speed = 2*pow((0.07+0.07)/10.0,2);

    for(int na = 0, nae = actions.size(); na < nae; ++na)
    {
        for (int i = 0, ie = pos_mesh.n_rows; i < ie; ++i)
        {
            assert(ie == 49);
            arma::vec mu(2);
            mu(MountainCar::StateLabel::position) = pos_mesh(i);
            mu(MountainCar::StateLabel::velocity) = vel_mesh(i);
            rewards.push_back(new mc_reward(mu,sigma_position, sigma_speed, na));
        }
        rewards.push_back(new mc_reward_one(na));
    }

    assert(rewards.size() == 150);

    IRLGradType atype = IRLGradType::RB;
    PlaneGIRL<FiniteAction,DenseState> pgirl(dataExpert, mlePolicy, rewards,
            mdp.getSettings().gamma, atype);
    cpu_timer timer2;
    timer2.start();
    pgirl.run();
    timer2.stop();
//    timefile << timer2.format(10, "%w") << std::endl;

    cout << "Weights (plane): " << pgirl.getWeights().t();



    /*** LSPI ***/
    e_GreedyApproximate lspiPolicy;
    lspiPolicy.setEpsilon(0);
    lspiPolicy.setNactions(actions.size());
//    LSPI<FiniteAction> lspi(lspiData,lspiPolicy, mdp.getSettings().gamma);


    return 0;
}
