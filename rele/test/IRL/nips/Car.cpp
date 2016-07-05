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

#include "rele/IRL/ParametricRewardMDP.h"

#include "rele/utils/FileManager.h"

#include "rele/approximators/regressors/others/LinearApproximator.h"
#include "rele/approximators/features/DenseFeatures.h"
#include "rele/approximators/features/TilesCoder.h"
#include "rele/approximators/tiles/BasicTiles.h"

#include "rele/algorithms/batch/td/LSPI.h"

#include "rele/environments/CarOnHill.h"

#include "rele/statistics/DifferentiableNormals.h"
#include "rele/policy/q_policy/e_Greedy.h"
#include "rele/policy/parametric/differentiable/GenericGibbsPolicy.h"

#include "rele/IRL/algorithms/EGIRL.h"
#include "rele/IRL/algorithms/CurvatureEGIRL.h"
#include "rele/IRL/algorithms/SDPEGIRL.h"

using namespace std;
using namespace ReLe;
using namespace arma;

template<class StateC>
class ProbabilityPolicy : public DifferentiablePolicy<FiniteAction, StateC>
{
public:

    /**
     * Create an instance of the class using the given regressor.
     *
     * \param actionsN the number of actions
     * \param energy the energy function \f$Q(x,u,\theta)\f$
     * \param temperature the temperature value
     */
    ProbabilityPolicy(unsigned int actionsN,
                      ParametricRegressor& prob) :
        actionsN(actionsN),  approximator(prob)
    {
    }

    virtual ~ProbabilityPolicy()
    {
    }


    // Policy interface
public:
    std::string getPolicyName() override
    {
        return std::string("GenericParametricGibbsPolicy");
    }

    hyperparameters_map getPolicyHyperparameters() override
    {
        hyperparameters_map hyperParameters;
        return hyperParameters;
    }

    std::string printPolicy() override
    {
        return std::string("");
    }


    double operator() (typename state_type<StateC>::const_type_ref state,
                       const unsigned int& action) override
    {
        int statesize = state.size();

        arma::vec tuple(1+statesize);
        tuple(arma::span(0, statesize-1)) = state;

        arma::vec&& distribution = computeDistribution(actionsN, tuple,	statesize);
        return distribution[action];
    }

    unsigned int operator() (typename state_type<StateC>::const_type_ref state) override
    {
        int statesize = state.size();

        arma::vec tuple(1+statesize);
        tuple(arma::span(0, statesize-1)) = state;


        arma::vec&& distribution = computeDistribution(actionsN, tuple,	statesize);
        return RandomGenerator::sampleDiscrete(distribution.begin(), distribution.end());
    }

    virtual ProbabilityPolicy<StateC>* clone() override
    {
        return new  ProbabilityPolicy<StateC>(*this);
    }

    // ParametricPolicy interface
public:
    virtual inline arma::vec getParameters() const override
    {
        return approximator.getParameters();
    }

    virtual inline const unsigned int getParametersSize() const override
    {
        return approximator.getParametersSize();
    }

    virtual inline void setParameters(const arma::vec& w) override
    {
        approximator.setParameters(w);
    }

    // DifferentiablePolicy interface
public:
    virtual arma::vec diff(typename state_type<StateC>::const_type_ref state,
                           typename action_type<FiniteAction>::const_type_ref action) override
    {
        ProbabilityPolicy& pi = *this;
        return pi(state, action)*difflog(state, action);
    }

    virtual arma::vec difflog(typename state_type<StateC>::const_type_ref state,
                              typename action_type<FiniteAction>::const_type_ref action) override
    {
        return arma::vec();
    }

    virtual arma::mat diff2log(typename state_type<StateC>::const_type_ref state,
                               typename action_type<FiniteAction>::const_type_ref action) override
    {
        return arma::mat();
    }

private:
    arma::vec computeDistribution(int nactions, arma::vec tuple, int statesize)
    {
        int na_red = nactions - 1;
        arma::vec distribution(nactions);
        distribution(na_red) = 1.0; //last action is valued 1.0
        double den = 1.0; //set the value of the last action to the den
        for (unsigned int k = 0; k < na_red; k++)
        {
            tuple[statesize] = k;
            double val = as_scalar(approximator(tuple));
            den += val;
            distribution[k] = val;
        }

        // check extreme cases (if some action is nan or infinite)
        arma::uvec q_nf = arma::find_nonfinite(distribution);
        if (q_nf.n_elem > 0)
        {
            arma::uvec q_f = arma::find_finite(distribution);
            distribution(q_f).zeros();
            distribution(q_nf).ones();
            den = q_nf.n_elem;
        }

        distribution /= den;
        return distribution;
    }

protected:
    unsigned int actionsN;
    ParametricRegressor& approximator;

};


int main(int argc, char *argv[])
{
    FileManager fm("nips", "car");
    fm.createDir();
    fm.cleanDir();

    // Define domain
    CarOnHill mdp;

    // Define linear regressors
    unsigned int tilesN = 15;
    unsigned int actionsN = mdp.getSettings().actionsNumber;
    Range xRange(-1, 1);
    Range vRange(-3, 3);

    auto* tiles = new BasicTiles({xRange, vRange, Range(-0.5, 1.5)}, {tilesN, tilesN, actionsN});

    DenseTilesCoder qphi(tiles);

    LinearApproximator linearQ(qphi);


    //Define solver
    double epsilon = 1e-6;
    LSPI batchAgent(linearQ, epsilon);

    e_GreedyApproximate expertPolicy;
    batchAgent.setPolicy(expertPolicy);

    //Run experiments and learning
    auto&& core = buildBatchCore(mdp, batchAgent);

    core.getSettings().episodeLength = mdp.getSettings().horizon;
    core.getSettings().nEpisodes = 1000;
    core.getSettings().maxBatchIterations = 30;
    core.getSettings().datasetLogger = new WriteBatchDatasetLogger<FiniteAction, DenseState>(fm.addPath("car.log"));
    core.getSettings().agentLogger = new BatchAgentPrintLogger<FiniteAction, DenseState>();

    core.run(1);


    expertPolicy.setEpsilon(0.0);

    core.getSettings().nEpisodes = 1;
    auto&& dataOptimal = core.runTest();

    std::cout << std::endl << "--- Running Test episode ---" << std::endl << std::endl;
    dataOptimal.printDecorated(std::cout);


    //Create expert distribution
    auto* tilesP = new BasicTiles({xRange, vRange}, {tilesN, tilesN});

    DenseTilesCoder pphi(tilesP);
    LinearApproximator energyApproximator(pphi);
    /*GenericParametricGibbsPolicyAllPref<DenseState> policyFamily(mdp.getSettings().actionsNumber,
    			energyApproximator, 0.01);*/


    ProbabilityPolicy<DenseState> policyFamily(mdp.getSettings().actionsNumber, energyApproximator);


    vec preferences = exp(linearQ.getParameters()/0.01);
    vec preferences1 = preferences.rows(0, preferences.n_rows/2 -1);
    vec preferences2 = preferences.rows(preferences.n_rows/2,  preferences.n_rows -1);



    vec muExpert = preferences1/(preferences1+preferences2);
    mat SigmaExpert = 1e-2*eye(muExpert.n_elem, muExpert.n_elem);
    ParametricNormal expertDist(muExpert, SigmaExpert);


    //policyFamily.setParameters(muExpert);
    //PolicyEvalAgent<FiniteAction, DenseState> expert(policyFamily);
    //Generate dataset from expert distribution
    PolicyEvalDistribution<FiniteAction, DenseState> expert(expertDist, policyFamily, 10);

    Core<FiniteAction, DenseState> expertCore(mdp, expert);
    CollectorStrategy<FiniteAction, DenseState> collection;
    expertCore.getSettings().loggerStrategy = &collection;
    expertCore.getSettings().episodeLength = mdp.getSettings().horizon;
    expertCore.getSettings().testEpisodeN = 1000;
    expertCore.runTestEpisodes();
    Dataset<FiniteAction,DenseState>& data = collection.data;

    std::cout << std::endl << "--- Running Test episode ---" << std::endl << std::endl;
    //data.printDecorated(cout);
    cout << "performances distribution: " << data.getMeanReward(mdp.getSettings().gamma) << endl;

    // Create parametric reward
    unsigned tilesRewardN = 15;
    auto* rewardTiles = new BasicTiles({xRange, vRange}, {tilesRewardN, tilesRewardN});
    DenseTilesCoder phiReward(rewardTiles);
    LinearApproximator rewardRegressor(phiReward);

    //Create IRL algorithm to run
    arma::mat theta = expert.getParams();
    /*auto* irlAlg = new EGIRL<FiniteAction, DenseState>(data, theta, expertDist,
            rewardRegressor, mdp.getSettings().gamma, IrlEpGrad::PGPE_BASELINE);*/
    /*auto* irlAlg = new CurvatureEGIRL<FiniteAction, DenseState>(data, theta, expertDist,
                rewardRegressor, mdp.getSettings().gamma, IrlEpGrad::PGPE_BASELINE, IrlEpHess::PGPE_BASELINE);*/
    auto* irlAlg = new SDPEGIRL<FiniteAction, DenseState>(data, theta, expertDist,
            rewardRegressor, mdp.getSettings().gamma, IrlEpGrad::PGPE_BASELINE, IrlEpHess::PGPE_BASELINE);


    //Run GIRL
    irlAlg->run();
    arma::vec omega = rewardRegressor.getParameters();
    omega.save(fm.addPath("Weights.txt"),  arma::raw_ascii);

    //Learn back environment
    ParametricRewardMDP<FiniteAction, DenseState> prMdp(mdp, rewardRegressor);
    batchAgent.setPolicy(expertPolicy);

    //Run experiments and learning
    expertPolicy.setEpsilon(1.0);
    auto&& imitatorCore = buildBatchCore(prMdp, batchAgent);

    imitatorCore.getSettings().episodeLength = mdp.getSettings().horizon;
    imitatorCore.getSettings().nEpisodes = 1000;
    imitatorCore.getSettings().maxBatchIterations = 30;
    //imitatorCore.getSettings().datasetLogger = new WriteBatchDatasetLogger<FiniteAction, DenseState>(fm.addPath("car.log"));
    imitatorCore.getSettings().agentLogger = new BatchAgentPrintLogger<FiniteAction, DenseState>();

    imitatorCore.run(1);

    expertPolicy.setEpsilon(0.0);

    core.getSettings().nEpisodes = 1;
    auto&& dataImitator = core.runTest();

    dataImitator.printDecorated(cout);
    cout << "imitator performance: " << dataImitator.getMeanReward(mdp.getSettings().gamma) << endl;



    return 0;
}
