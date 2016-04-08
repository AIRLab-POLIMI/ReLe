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

#include "rele/algorithms/policy_search/PGPE/PGPE.h"
#include "rele/algorithms/policy_search/NES/NES.h"
#include "rele/algorithms/policy_search/REPS/REPS.h"
#include "rele/statistics/DifferentiableNormals.h"
#include "rele/core/Core.h"
#include "rele/policy/parametric/differentiable/NormalPolicy.h"
#include "rele/approximators/basis/IdentityBasis.h"
#include "rele/approximators/features/DenseFeatures.h"
#include "rele/utils/RandomGenerator.h"
#include "rele/utils/FileManager.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <random>
#include <cmath>
#include "rele/environments/NLS.h"

using namespace std;
using namespace ReLe;
using namespace arma;

struct bboConfig
{
    unsigned int nbRuns, nbPolicies;
    double stepLength;
    GradientStep* steprule = nullptr;

    virtual ~bboConfig()
    {
        if (steprule != nullptr)
            delete steprule;
    }
};

void help()
{
    cout << "nls_BBO algorithm [] #Updates #Policies stepLength [updaterule]" << endl;
    cout << " - algorithm: pgpe, nes, enes, reps" << endl;
    cout << "   * notes that pgpe and nes requires and additional paramiter" << endl;
    cout << "   * distribution: gauss, log, chol, diag" << endl;
    cout << " - updaterule: 'constant', 'adaptive' (default)" << endl;
}

bool InputValidation(int argc, char *argv[], bboConfig& config, int offset)
{
    if (argc < 5+offset)
    {
        std::cout << "ERROR: Too few arguments." << endl;
        help();
        return false;
    }

    int nbRuns         = atoi(argv[2+offset]);
    int nbPolicies     = atoi(argv[3+offset]);
    double step_length = atof(argv[4+offset]);

    // check arguments
    if (nbRuns < 1 || nbPolicies < 1 || step_length <= 0)
    {
        std::cout << "ERROR: Arguments not valid\n";
        return false;
    }


    if (argc == 6+offset)
    {
        if (strcmp(argv[5+offset], "constant") == 0)
        {
            config.steprule = new ConstantGradientStep(step_length);
        }
        else if (strcmp(argv[5+offset], "adaptive") == 0)
        {
            config.steprule = new AdaptiveGradientStep(step_length);
        }
        else
        {
            std::cout << "ERROR: Arguments not valid\n";
            return false;
        }
    }
    else
    {
        config.steprule = new AdaptiveGradientStep(step_length);
    }

    // load valid arguments in the configuration
    config.nbRuns      = nbRuns;
    config.nbPolicies  = nbPolicies;
    config.stepLength  = step_length;

    return true;
}

/**
 *
 * argv[1] learning algorithm name (pgpe, nes, enes, reps) -> pgpe and nes requires the distribution type
 * argv[2] distribution type (gauss, chol, diag, log)
 * argv[2/3] # updates
 * argv[3/4] # policies per update
 * argv[4/5] learning rate for updates
 * argv[5/6] stepType ("constant", "adaptive")
 *
 */
int main(int argc, char *argv[])
{
    bboConfig config;

    //--- INPUT VALIDATION
    char alg[10], polType[10];
    if (argc > 1)
    {
        strncpy(alg, argv[1], 10);
        int offset = 0;
        if ((strcmp(alg,"pgpe") == 0) || (strcmp(alg,"nes") == 0))
        {
            if (argc > 2)
            {
                strncpy(polType, argv[2], 10);
                offset = 1;
            }
            else
            {
                std::cout << "ERROR: Too few arguments." << endl;
                help();
                exit(1);
            }
        }
        else if (strcmp(alg,"reps") == 0)
        {
            strcpy(polType, "gauss");
        }
        else if (strcmp(alg,"enes") == 0)
        {
            strcpy(polType, "chol");
        }
        else
        {
            cout << "Unknown algorithm!!" << endl;
            help();
            exit(1);
        }

        // load the arguments in the configuration
        if ( ! InputValidation(argc, argv, config, offset) )
        {
            // if the arguments are not valid then the application ends
            return -1;
        }
    }
    else
    {
        // default configuration if no arguments are specified
        strcpy(alg, "nes");
        strcpy(polType, "diag");
        config.nbRuns      = 400;
        config.nbPolicies  = 100;
        config.stepLength  = 0.01;
        config.steprule = new AdaptiveGradientStep(config.stepLength);
    }
    //---

    FileManager fm("nls", "BBO");
    fm.createDir();
    //    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);

    NLS mdp;
    //with these settings
    //max in ( many optimal points ) -> J = 8.5
    //note that there are multiple optimal solutions
    //e.g.
    //-3.2000    8.8000    8.4893
    //-3.2000    9.3000    8.4959
    //-3.2000    9.5000    8.4961
    //-3.4000   10.0000    8.5007
    //-3.2000    9.4000    8.5020
    //-3.1000    8.8000    8.5028
    //-3.4000    9.7000    8.5041
    //-3.0000    8.1000    8.5205
    //-2.9000    7.7000    8.5230
    //-3.1000    9.1000    8.5243
    //-2.8000    7.3000    8.5247

    int dim = mdp.getSettings().stateDimensionality;

    //--- define policy (low level)
    BasisFunctions basis = IdentityBasis::generate(dim);
    DenseFeatures phi(basis);

    BasisFunctions stdBasis = IdentityBasis::generate(dim);
    DenseFeatures stdPhi(stdBasis);
    arma::vec stdWeights(stdPhi.rows());
    stdWeights.fill(0.5);

    NormalStateDependantStddevPolicy policy(phi, stdPhi, stdWeights);
    //---

    //--- distribution setup
    int nparams = phi.rows();
    arma::vec mean(nparams, fill::zeros);
    mean[0] = -0.42;
    mean[1] =  0.42;
    DifferentiableDistribution* dist;

    if (strcmp(polType, "gauss") == 0)
    {
        //----- ParametricNormal
        arma::mat cov(nparams, nparams, arma::fill::eye);
        dist = new ParametricNormal(mean, cov);
    }
    else if (strcmp(polType, "log") == 0)
    {
        //----- ParametricLogisticNormal
        dist = new ParametricLogisticNormal(mean, zeros(nparams), 1);
    }
    else if (strcmp(polType, "chol") == 0)
    {
        //----- ParametricCholeskyNormal
        arma::mat cov(nparams, nparams, arma::fill::eye);
        mat cholMtx = chol(cov);
        dist = new ParametricCholeskyNormal(mean, cholMtx);
    }
    else if (strcmp(polType, "diag") == 0)
    {
        //----- ParametricDiagonalNormal
        vec sigmas(nparams, fill::ones);
        dist = new ParametricDiagonalNormal(mean, sigmas);
        //-----
    }
    //---

    cout << "## MetaDistribution: " << dist->getDistributionName() << endl;

    int nbepperpol = 1, nbpolperupd = config.nbPolicies;
    char outputname[100];
    ReLe::Core<DenseAction, DenseState>* core;
    if (strcmp(alg, "pgpe") == 0)
    {
        bool usebaseline = true;
        PGPE<DenseAction, DenseState>* agent = new PGPE<DenseAction, DenseState>
        (*dist, policy, nbepperpol, nbpolperupd, *(config.steprule), usebaseline);
        core = new ReLe::Core<DenseAction, DenseState>(mdp, *agent);
        sprintf(outputname, "nls_pgpe_%s.log", polType);
    }
    else if (strcmp(alg, "nes") == 0)
    {
        bool usebaseline = true;
        NES<DenseAction, DenseState>* agent = new NES<DenseAction, DenseState>
        (*dist, policy, nbepperpol, nbpolperupd, *(config.steprule), usebaseline);
        core = new ReLe::Core<DenseAction, DenseState>(mdp, *agent);
        sprintf(outputname, "nls_nes_%s.log", polType);
    }
    else if (strcmp(alg, "enes") == 0)
    {
        bool usebaseline = true;
        arma::vec mean(nparams, fill::zeros);
        arma::mat cov(nparams, nparams, arma::fill::eye);
        mat cholMtx = chol(cov);
        ParametricCholeskyNormal* distr = new ParametricCholeskyNormal(mean, cholMtx);
        eNES<DenseAction, DenseState>* agent= new eNES<DenseAction, DenseState>
        (*distr, policy, nbepperpol, nbpolperupd, *(config.steprule), usebaseline);
        core = new ReLe::Core<DenseAction, DenseState>(mdp, *agent);
        sprintf(outputname, "nls_enes.log");
    }
    else if (strcmp(alg, "reps") == 0)
    {
        arma::vec mean(nparams, fill::zeros);
        arma::mat cov(nparams, nparams, arma::fill::eye);
        ParametricNormal* distr= new ParametricFullNormal(mean, cov);
        REPS<DenseAction, DenseState>* agent = new REPS<DenseAction, DenseState>(*distr,policy,nbepperpol,nbpolperupd);
        agent->setEps(0.9);
        core = new ReLe::Core<DenseAction, DenseState>(mdp, *agent);
        sprintf(outputname, "nls_reps.log");
    }
    else
    {
        std::cout << "ERROR: Algorithm " << alg << " not found in (pgpe, nes, enes, reps)\n";
        abort();
    }

    WriteStrategy<DenseAction, DenseState> wStrategy(
        fm.addPath(outputname),
        WriteStrategy<DenseAction, DenseState>::AGENT,
        true /*delete file*/
    );
    core->getSettings().loggerStrategy = &wStrategy;

    int horiz = mdp.getSettings().horizon;
    core->getSettings().episodeLength = horiz;

    int nbUpdates = config.nbRuns;
    int episodes  = nbUpdates*nbepperpol*nbpolperupd;
    double every, bevery;
    every = bevery = 0.1; //%
    int updateCount = 0;
    for (int i = 0; i < episodes; i++)
    {
        core->runEpisode();

        int v = nbepperpol*nbpolperupd;
        if (i % v == 0)
        {
            updateCount++;
            if ((updateCount >= nbUpdates*every) || (updateCount == 1))
            {
                int p = 100 * updateCount/static_cast<double>(nbUpdates);
                cout << "### " << p << "% ###" << endl;
                //                cout << dist->getParameters().t();
                core->getSettings().testEpisodeN = 100;
                arma::vec J = core->runEvaluation();
                cout << "mean score: " << J(0) << endl;
                every += bevery;
            }
        }
    }

    //    int nbTestEpisodes = 1000;
    //    cout << "Final test [#episodes: " << nbTestEpisodes << " ]" << endl;
    //    core->getSettings().testEpisodeN = nbTestEpisodes;
    //    cout << core->runBatchTest() << endl;

    //    //--- collect some trajectories
    //    core->getSettings().loggerStrategy = new WriteStrategy<DenseAction, DenseState>(
    //        fm.addPath("NlsFinal.log"),
    //        WriteStrategy<DenseAction, DenseState>::TRANS,
    //        true /*delete file*/
    //    );
    //    for (int n = 0; n < 100; ++n)
    //        core->runTestEpisode();
    //    //---

    delete dist;
    delete core;
    return 0;
}
