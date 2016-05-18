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

#include "rele/core/Core.h"
#include "rele/core/BatchCore.h"
#include "rele/core/PolicyEvalAgent.h"

#include "rele/environments/MountainCar.h"

#include "rele/statistics/DifferentiableNormals.h"
#include "rele/policy/nonparametric/RandomPolicy.h"

#include "rele/approximators/basis/IdentityBasis.h"
#include "rele/approximators/basis/GaussianRbf.h"
#include "rele/approximators/basis/ConditionBasedFunction.h"
#include "rele/approximators/features/DenseFeatures.h"
#include "rele/approximators/regressors/others/LinearApproximator.h"

#include "rele/algorithms/batch/td/LSPI.h"

#include "rele/IRL/ParametricRewardMDP.h"
#include "rele/IRL/algorithms/EGIRL.h"

#include "rele/utils/FileManager.h"

#include <boost/timer/timer.hpp>

using namespace boost::timer;
using namespace std;
using namespace ReLe;
using namespace arma;

class mountain_car_manual_policy : public ParametricPolicy<FiniteAction, DenseState>
{
public:
    unsigned int operator()(const arma::vec& state) override
    {
        if (RandomGenerator::sampleEvent(0.4))
        {
            return RandomGenerator::sampleUniformInt(0,1);
        }
        else
        {
            double speed = state(MountainCar::StateLabel::velocity);
            if (speed <= eps)
                return 0;
            else
                return 2;
        }
    }

    double operator()(const arma::vec& state, const unsigned int& action) override
    {
        double speed = state(MountainCar::StateLabel::velocity);
        if (speed <= eps && action == 0)
            return 1;
        else if(speed > eps && action == 2)
            return 1;
        else
            return 0;
    }

    //Parametric policy interface
public:
    virtual arma::vec getParameters() const override
    {
        arma::vec w = {eps};

        return w;
    }

    virtual const unsigned int getParametersSize() const override
    {
        return 1;
    }

    virtual void setParameters(const arma::vec& w) override
    {
        eps = w(0);
        std::cout << eps << std::endl;
    }


    // Policy interface
public:
    string getPolicyName() override
    {
        return "mountain_car_manual_policy";
    }

    string printPolicy() override
    {
        return "";
    }

    mountain_car_manual_policy* clone() override
    {
        return new mountain_car_manual_policy();
    }

private:
    double eps;
};

int main(int argc, char *argv[])
{
    RandomGenerator::seed(4265436624);

    FileManager fm("mc", "GIRL");
    fm.createDir();
    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);

    int nbEpisodes = 100;  // number of episodes simulated to generate mle data
    int budget = 300;      // number of samples used to estimate policy and learn weights
    budget = 4000;

    // define domain
    MountainCar mdp(MountainCar::ConfigurationsLabel::Klein);

    // === define expert's policy === //
    mountain_car_manual_policy expertPolicy;

    arma::vec muExpert = {0};
    arma::mat SigmaExpert = { 1e-3 };
    ParametricNormal expertDist(muExpert, SigmaExpert);


    // === get expert's trajectories === //
    PolicyEvalDistribution<FiniteAction, DenseState> expert(expertDist, expertPolicy);
    Core<FiniteAction, DenseState> expertCore(mdp, expert);
    CollectorStrategy<FiniteAction, DenseState> collection;
    expertCore.getSettings().loggerStrategy = &collection;
    expertCore.getSettings().episodeLength = mdp.getSettings().horizon;
    expertCore.getSettings().testEpisodeN = nbEpisodes;
    expertCore.runTestEpisodes();

    auto& data = collection.data;
    auto theta = expert.getParams();

    // === get only trailing info === //
    Dataset<FiniteAction,DenseState> dataExpert;
    for (int ep = 0; ep < data.size() && budget > 0; ++ep)
    {
        Episode<FiniteAction,DenseState> episodeExpert;
        int nbSamples = data[ep].size();
        for (int t = 0; t < nbSamples && budget > 0; ++t)
        {
            episodeExpert.push_back(data[ep][t]);
            --budget;
        }
        dataExpert.push_back(episodeExpert);
    }

    arma::mat thetaExpert = theta.cols(0, dataExpert.size()-1);

    std::cout << "n episodes: " << dataExpert.size() << std::endl;
    std::cout << "thetaExpert: " << thetaExpert.n_rows << "x" << thetaExpert.n_cols  << std::endl;


    // === recover reward by IRL === //

    vec pos_linspace = linspace(-1.2,0.6,7);
    vec vel_linspace = linspace(-0.07,0.07,7);

    arma::mat yy_vel, xx_pos;
    meshgrid(vel_linspace, pos_linspace, yy_vel, xx_pos);

    arma::vec pos_mesh = vectorise(xx_pos);
    arma::vec vel_mesh = vectorise(yy_vel);
    arma::mat XX = arma::join_horiz(vel_mesh,pos_mesh);


    double sigma_position = 2*pow((0.6+1.2)/4.0,2);
    double sigma_speed    = 2*pow((0.07+0.07)/4.0,2);


    arma::vec widths = {sigma_speed, sigma_position};
    arma::mat WW = repmat(widths, 1, XX.n_rows);
    arma::mat XT = XX.t();

    BasisFunctions rewardBasis = GaussianRbf::generate(XT, WW);
    DenseFeatures phiReward(rewardBasis);
    LinearApproximator rewardF(phiReward);

    cout << "Rewards size: " << rewardF.getParametersSize() << endl;

    EGIRL<FiniteAction,DenseState> irlAlg(dataExpert, thetaExpert, expertDist, rewardF,
                                          mdp.getSettings().gamma, IrlEpGrad::PGPE_BASELINE);

    irlAlg.run();

    vec rewWeights = rewardF.getParameters();
    cout << "Weights (EGIRL): " << rewWeights.t();

    // === LSPI === //
    //define basis for Q-function
    /*vector<FiniteAction> actions;
    for (int i = 0; i < mdp.getSettings().actionsNumber; ++i)
        actions.push_back(FiniteAction(i));
    BasisFunctions qbasis = GaussianRbf::generate(XT, WW);
    BasisFunctions qbasisrep = AndConditionBasisFunction::generate(qbasis, 2, actions.size());
    //create basis vector
    DenseFeatures qphi(qbasisrep);
    StochasticDiscretePolicy<FiniteAction,DenseState> randomPolicy(actions);

    //create data per lspi
    PolicyEvalAgent<FiniteAction, DenseState> lspiEval(randomPolicy);
    Core<FiniteAction, DenseState> lspiCore(mdp, lspiEval);
    CollectorStrategy<FiniteAction, DenseState> collectionlspi;
    lspiCore.getSettings().loggerStrategy = &collectionlspi;
    lspiCore.getSettings().episodeLength = 100;
    lspiCore.getSettings().testEpisodeN = 1000;
    lspiCore.runTestEpisodes();


    Dataset<FiniteAction,DenseState>& dataLSPI = collectionlspi.data;

    e_GreedyApproximate lspiPolicy;
    lspiPolicy.setEpsilon(0.01);
    lspiPolicy.setNactions(actions.size());

    LSPI<FiniteAction> batchAgent(dataLSPI, lspiPolicy, qphi, 0.01);

    auto&& core = buildBatchOnlyCore(dataLSPI, batchAgent);

    core.getSettings().maxBatchIterations = 100;

    double gamma = 0.9;
    EnvironmentSettings envSettings;
    envSettings.gamma = gamma;

    core.run(envSettings);

    lspiPolicy.setEpsilon(0.0);

    //create data per lspi
    PolicyEvalAgent<FiniteAction, DenseState> finalEval(lspiPolicy);
    Core<FiniteAction, DenseState> finalCore(mdp, finalEval);
    CollectorStrategy<FiniteAction, DenseState> collectionFinal;
    finalCore.getSettings().loggerStrategy = &collectionFinal;
    finalCore.getSettings().episodeLength = 150;
    finalCore.getSettings().testEpisodeN = 1000;
    finalCore.runTestEpisodes();


    // === save data === //
    Dataset<FiniteAction,DenseState>& dataFinal = collectionFinal.data;


    // output expert
    double meanLenghtExpert = 0;
    for(auto& ep : dataExpert)
    {
        meanLenghtExpert += ep.size();
    }

    meanLenghtExpert /= dataExpert.size();


    std::cout << "Expert max length: " << dataExpert.getEpisodeMaxLength() << std::endl;
    std::cout << "Expert mean length: " << meanLenghtExpert << std::endl;


    // output data results
    double meanLenght = 0;
    for(auto& ep : data)
    {
        meanLenght += ep.size();
    }

    meanLenght /= data.size();


    std::cout << "Episode max length: " << dataFinal.getEpisodeMaxLength() << std::endl;
    std::cout << "Episode mean length: " << meanLenght << std::endl;*/

    return 0;
}
