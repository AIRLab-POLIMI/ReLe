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

#include "policy_search/gradient/onpolicy/REINFORCEAlgorithm.h"
#include "policy_search/gradient/onpolicy/GPOMDPAlgorithm.h"
#include "policy_search/gradient/onpolicy/NaturalPGAlgorithm.h"
#include "policy_search/gradient/onpolicy/ENACAlgorithm.h"
#include "DifferentiableNormals.h"
#include "Core.h"
#include "parametric/differentiable/NormalPolicy.h"
#include "parametric/differentiable/GenericNormalPolicy.h"
#include "regressors/SaturatedRegressor.h"
#include "basis/GaussianRbf.h"
#include "basis/SubspaceBasis.h"
#include "basis/ModularBasis.h"
#include "basis/NormBasis.h"
#include "features/SparseFeatures.h"
#include "RandomGenerator.h"
#include "FileManager.h"
#include "Pursuer.h"


#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <random>
#include <cmath>

using namespace std;
using namespace ReLe;
using namespace arma;


struct gradConfig
{
    unsigned int nbRuns, nbEpisodes;
    double stepLength;
    StepRule* steprule;

    virtual ~gradConfig()
    {
        delete steprule;
    }
};

struct WallNearBasis: public BasisFunction
{
public:
	enum dir {N, S, W, E};

public:
	WallNearBasis(dir wall, double threshold) : wall(wall), threshold(threshold)
	{

	}

    virtual double operator()(const arma::vec& input)
    {
    	switch(wall)
    	{
    		case N:
    			return abs(input[Pursuer::y] - 10) < threshold;

    		case S:
    			return abs(input[Pursuer::y] + 10) < threshold;

    		case W:
    			return abs(input[Pursuer::x] + 10) < threshold;

    		case E:
    			return abs(input[Pursuer::x] + 10) < threshold;

    		default:
    			return 0;
    	}

    }

    virtual void writeOnStream(std::ostream& out)
    {

    }

    virtual void readFromStream(std::istream& in)
    {

    }

    ~WallNearBasis()
    {

    }

private:
    dir wall;
    double threshold;
};

class PursuerDirectionBasis: public BasisFunction
{
public:
	virtual double operator()(const arma::vec& input)
	{
		return RangePi::wrap(atan2(input[Pursuer::yp], input[Pursuer::xp]));
    }

    virtual void writeOnStream(std::ostream& out)
    {

    }

    virtual void readFromStream(std::istream& in)
    {

    }

    virtual ~PursuerDirectionBasis()
    {

    }
};

void help()
{
    cout << "pursuer_PG algorithm #Updates #Episodes stepLength [updaterule]" << endl;
    cout << " - algorithm: r, rb, g, gb" << endl;
    cout << " - updaterule: 'constant', 'adaptive' (default)" << endl;
}

bool InputValidation(int argc, char *argv[], gradConfig& config)
{
    if (argc < 5)
    {
        std::cout << "ERROR: Too few arguments." << endl;
        help();
        return false;
    }

    int nbRuns         = atoi(argv[2]);
    int nbEpisodes     = atoi(argv[3]);
    double step_length = atof(argv[4]);

    // check arguments
    if (nbRuns < 1 || nbEpisodes < 1 || step_length <= 0)
    {
        std::cout << "ERROR: Arguments not valid\n";
        return false;
    }


    if (argc == 6)
    {
        if (strcmp(argv[5], "constant") == 0)
        {
            config.steprule = new ConstantStep(step_length);
        }
        else if (strcmp(argv[5], "adaptive") == 0)
        {
            config.steprule = new AdaptiveStep(step_length);
        }
        else
        {
            std::cout << "ERROR: Arguments not valid\n";
            return false;
        }
    }
    else
    {
        config.steprule = new AdaptiveStep(step_length);
    }

    // load valid arguments in the configuration
    config.nbRuns      = nbRuns;
    config.nbEpisodes  = nbEpisodes;
    config.stepLength  = step_length;

    return true;
}

/**
 *
 * argv[1] learning algorithm name (r,g,rb,gb,gsb,n,nb)
 * argv[2] # updates
 * argv[3] # episodes per update
 * argv[4] learning rate for updates
 * argv[5] stepType ("constant", "adaptive")
 *
 */
int main(int argc, char *argv[])
{

    gradConfig config;

    //--- INPUT VALIDATION
    char alg[10];
    if (argc > 1)
    {
        strncpy(alg, argv[1], 10);

        // load the arguments in the configuration
        if ( ! InputValidation(argc, argv, config) )
        {
            // if the arguments are not valid then the application ends
            return -1;
        }
    }
    else
    {
        // default configuration if no arguments are specified
        strcpy(alg, "r");
        config.nbRuns      = 400;
        config.nbEpisodes  = 100;
        config.stepLength  = 0.01;
        config.steprule    = new AdaptiveStep(config.stepLength);
    }
    //---

    FileManager fm("pursuer", "PG");
    fm.createDir();
    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);

    Pursuer mdp;

    int dim = mdp.getSettings().continuosStateDim;

    //--- define policy (low level)
    BasisFunctions basis;

    basis.push_back(new SubspaceBasis(new NormBasis(), arma::span(Pursuer::xp, Pursuer::yp)));
    basis.push_back(new SubspaceBasis(new NormBasis(), arma::span(Pursuer::x, Pursuer::y)));
    basis.push_back(new ModularDifference(Pursuer::theta, Pursuer::thetap, RangePi()));
    basis.push_back(new PursuerDirectionBasis());
    double criticalDistance = 0.5;
    basis.push_back(new WallNearBasis(WallNearBasis::N, criticalDistance));
    basis.push_back(new WallNearBasis(WallNearBasis::S, criticalDistance));
    basis.push_back(new WallNearBasis(WallNearBasis::W, criticalDistance));
    basis.push_back(new WallNearBasis(WallNearBasis::E, criticalDistance));



    SparseFeatures phi(basis, 2);

    arma::mat cov(2, 2, arma::fill::eye);
    cov *= 0.1;
    MVNPolicy policy(phi, cov);
    //SaturatedRegressor regressor(phi, {0, -M_PI}, {1, M_PI});
    //GenericMVNPolicy policy(regressor, cov);
    //---

    AbstractPolicyGradientAlgorithm<DenseAction, DenseState>* agent;
    int nbepperpol = config.nbEpisodes;
    unsigned int rewardId = 0;
    char outputname[100];
    if (strcmp(alg, "r"  ) == 0)
    {
        cout << "REINFORCEAlgorithm" << endl;
        bool usebaseline = false;
        agent = new REINFORCEAlgorithm<DenseAction, DenseState>(policy, nbepperpol,
                *(config.steprule), usebaseline, rewardId);
        sprintf(outputname, "pursuer_r.log");
    }
    else if (strcmp(alg, "g"  ) == 0)
    {
        cout << "GPOMDPAlgorithm" << endl;
        agent = new GPOMDPAlgorithm<DenseAction, DenseState>(policy, nbepperpol,
                mdp.getSettings().horizon, *(config.steprule), rewardId);
        sprintf(outputname, "pursuer_g.log");
    }
    else if (strcmp(alg, "rb" ) == 0)
    {
        cout << "REINFORCEAlgorithm BASELINE" << endl;
        bool usebaseline = true;
        agent = new REINFORCEAlgorithm<DenseAction, DenseState>(policy, nbepperpol,
                *(config.steprule), usebaseline, rewardId);
        sprintf(outputname, "pursuer_rb.log");
    }
    else if (strcmp(alg, "gb" ) == 0)
    {
        cout << "GPOMDPAlgorithm BASELINE" << endl;
        agent = new GPOMDPAlgorithm<DenseAction, DenseState>(policy, nbepperpol,
                mdp.getSettings().horizon, *(config.steprule),
                GPOMDPAlgorithm<DenseAction, DenseState>::BaseLineType::MULTI,
                rewardId);
        sprintf(outputname, "pursuer_gb.log");
    }
    else if (strcmp(alg, "gsb") == 0)
    {
        cout << "GPOMDPAlgorithm SINGLE BASELINE" << endl;
        agent = new GPOMDPAlgorithm<DenseAction, DenseState>(policy, nbepperpol,
                mdp.getSettings().horizon, *(config.steprule),
                GPOMDPAlgorithm<DenseAction, DenseState>::BaseLineType::SINGLE,
                rewardId);
        sprintf(outputname, "pursuer_gsb.log");
    }
    else if (strcmp(alg, "natg") == 0)
    {
        cout << "NaturalGPOMDPAlgorithm BASELINE" << endl;
        bool usebaseline = true;
        agent = new NaturalGPOMDPAlgorithm<DenseAction, DenseState>(policy, nbepperpol,
                mdp.getSettings().horizon, *(config.steprule), usebaseline, rewardId);
        sprintf(outputname, "pursuer_natg.log");
    }
    else if (strcmp(alg, "natr") == 0)
    {
        cout << "NaturalREINFORCEAlgorithm BASELINE" << endl;
        bool usebaseline = true;
        agent = new NaturalREINFORCEAlgorithm<DenseAction, DenseState>(policy, nbepperpol,
                *(config.steprule), usebaseline, rewardId);
        sprintf(outputname, "pursuer_natr.log");
    }
    else if (strcmp(alg, "enac") == 0)
    {
        cout << "eNAC BASELINE" << endl;
        bool usebaseline = true;
        agent = new eNACAlgorithm<DenseAction, DenseState>(policy, nbepperpol,
                *(config.steprule), usebaseline, rewardId);
        sprintf(outputname, "pursuer_enac.log");
    }
    else
    {
        std::cout << "ERROR: Algorithm " << alg << " not found in (r, g, rb, gb, gsb, n, nb)\n";
        abort();
    }



    ReLe::Core<DenseAction, DenseState> core(mdp, *agent);
    core.getSettings().loggerStrategy = new WriteStrategy<DenseAction, DenseState>(
        fm.addPath(outputname), WriteStrategy<DenseAction, DenseState>::AGENT, true);

    core.getSettings().episodeLenght = mdp.getSettings().horizon;

    int nbUpdates = config.nbRuns;
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
                //cout << policy.getParameters().t();
                core.getSettings().testEpisodeN = 1000;
                arma::vec J = core.runBatchTest();
                cout << "mean score: " << J(0) << endl;
                if (updateCount != 1)
                    every += bevery;
            }
        }

        core.runEpisode();
    }

    delete core.getSettings().loggerStrategy;

    //--- collect some trajectories
    core.getSettings().loggerStrategy = new WriteStrategy<DenseAction, DenseState>(
        fm.addPath(outputname),  WriteStrategy<DenseAction, DenseState>::TRANS,false);
    core.getSettings().testEpisodeN = 3000;
    core.runTestEpisodes();
    //---

    cout << "Learned Parameters: " << endl;
    cout << policy.getParameters().t();

    delete core.getSettings().loggerStrategy;

    return 0;
}
