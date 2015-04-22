/*
 * rele,
 *
 *
 * Copyright (C) 2015  Davide Tateo & Matteo Pirotta
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

#include "policy_search/onpolicy/REINFORCEAlgorithm.h"
#include "policy_search/onpolicy/GPOMDPAlgorithm.h"
#include "policy_search/onpolicy/NaturalPGAlgorithm.h"
#include "policy_search/onpolicy/ENACAlgorithm.h"
#include "Core.h"
#include "parametric/differentiable/NormalPolicy.h"
#include "BasisFunctions.h"
#include "basis/PolynomialFunction.h"
#include "basis/GaussianRBF.h"
#include "RandomGenerator.h"
#include "FileManager.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <random>
#include <cmath>
#include "Dam.h"

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

void help()
{
    cout << "dam_PG algorithm #Updates #Episodes stepLength [updaterule]" << endl;
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

    FileManager fm("dam", "PG");
    fm.createDir();
//    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);

    Dam mdp;

    PolynomialFunction *pf = new PolynomialFunction();
    GaussianRbf* gf1 = new GaussianRbf(0, 50, true);
    GaussianRbf* gf2 = new GaussianRbf(50, 20, true);
    GaussianRbf* gf3 = new GaussianRbf(120, 40, true);
    GaussianRbf* gf4 = new GaussianRbf(160, 50, true);
    BasisFunctions basis;
    basis.push_back(pf);
    basis.push_back(gf1);
    basis.push_back(gf2);
    basis.push_back(gf3);
    basis.push_back(gf4);

    DenseFeatures phi(basis);
    MVNLogisticPolicy policy(phi, 50);
    vec p(6);
    p(0) = 50;
    p(1) = -50;
    p(2) = 0;
    p(3) = 0;
    p(4) = 50;
    p(5) = 0;
    policy.setParameters(p);


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
        sprintf(outputname, "dam_r.log");
    }
    else if (strcmp(alg, "g"  ) == 0)
    {
        cout << "GPOMDPAlgorithm" << endl;
        agent = new GPOMDPAlgorithm<DenseAction, DenseState>(policy, nbepperpol,
                mdp.getSettings().horizon, *(config.steprule), rewardId);
        sprintf(outputname, "dam_g.log");
    }
    else if (strcmp(alg, "rb" ) == 0)
    {
        cout << "REINFORCEAlgorithm BASELINE" << endl;
        bool usebaseline = true;
        agent = new REINFORCEAlgorithm<DenseAction, DenseState>(policy, nbepperpol,
                *(config.steprule), usebaseline, rewardId);
        sprintf(outputname, "dam_rb.log");
    }
    else if (strcmp(alg, "gb" ) == 0)
    {
        cout << "GPOMDPAlgorithm BASELINE" << endl;
        agent = new GPOMDPAlgorithm<DenseAction, DenseState>(policy, nbepperpol,
                mdp.getSettings().horizon, *(config.steprule),
                GPOMDPAlgorithm<DenseAction, DenseState>::BaseLineType::MULTI,
                rewardId);
        sprintf(outputname, "dam_gb.log");
    }
    else if (strcmp(alg, "gsb") == 0)
    {
        cout << "GPOMDPAlgorithm SINGLE BASELINE" << endl;
        agent = new GPOMDPAlgorithm<DenseAction, DenseState>(policy, nbepperpol,
                mdp.getSettings().horizon, *(config.steprule),
                GPOMDPAlgorithm<DenseAction, DenseState>::BaseLineType::SINGLE,
                rewardId);
        sprintf(outputname, "dam_gsb.log");
    }
    else if (strcmp(alg, "natg") == 0)
    {
        cout << "NaturalGPOMDPAlgorithm BASELINE" << endl;
        bool usebaseline = true;
        agent = new NaturalGPOMDPAlgorithm<DenseAction, DenseState>(policy, nbepperpol,
                mdp.getSettings().horizon, *(config.steprule), usebaseline, rewardId);
        sprintf(outputname, "dam_natg.log");
    }
    else if (strcmp(alg, "natr") == 0)
    {
        cout << "NaturalREINFORCEAlgorithm BASELINE" << endl;
        bool usebaseline = true;
        agent = new NaturalREINFORCEAlgorithm<DenseAction, DenseState>(policy, nbepperpol,
                *(config.steprule), usebaseline, rewardId);
        sprintf(outputname, "dam_natr.log");
    }
    else if (strcmp(alg, "enac") == 0)
    {
        cout << "eNAC BASELINE" << endl;
        bool usebaseline = true;
        agent = new eNACAlgorithm<DenseAction, DenseState>(policy, nbepperpol,
                *(config.steprule), usebaseline, rewardId);
        sprintf(outputname, "dam_enac.log");
    }
    else
    {
        std::cout << "ERROR: Algorithm " << alg << " not found in (r, g, rb, gb, gsb, n, nb)\n";
        abort();
    }


    ReLe::Core<DenseAction, DenseState> core(mdp, *agent);
    core.getSettings().loggerStrategy = new WriteStrategy<DenseAction, DenseState>(
        fm.addPath(outputname),
        WriteStrategy<DenseAction, DenseState>::AGENT,
        true /*delete file*/
    );

    int horiz = mdp.getSettings().horizon;
    core.getSettings().episodeLenght = horiz;

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
                cout << policy.getParameters().t();
                core.getSettings().testEpisodeN = 1000;
                arma::vec J = core.runBatchTest();
                cout << "mean score: " << J(0) << endl;
                if (updateCount != 1)
                    every += bevery;
            }
        }

        core.runEpisode();
    }

    //    int nbTestEpisodes = 1000;
    //    cout << "Final test [#episodes: " << nbTestEpisodes << " ]" << endl;
    //    core.getSettings().testEpisodeN = 1000;
    //    cout << core.runBatchTest() << endl;

    //    //--- collect some trajectories
    //    core.getSettings().loggerStrategy = new WriteStrategy<DenseAction, DenseState>(
    //        fm.addPath("NlsFinal.log"),
    //        WriteStrategy<DenseAction, DenseState>::TRANS,
    //        true /*delete file*/
    //    );
    //    for (int n = 0; n < 100; ++n)
    //        core.runTestEpisode();
    //    //---

    return 0;
}
