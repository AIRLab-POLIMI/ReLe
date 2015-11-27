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

#include "policy_search/NES/NES.h"
#include "policy_search/REPS/REPS.h"
#include "DifferentiableNormals.h"
#include "Core.h"
#include "parametric/differentiable/GibbsPolicy.h"
#include "BasisFunctions.h"
#include "basis/PolynomialFunction.h"
#include "basis/ConditionBasedFunction.h"
#include "basis/GaussianRbf.h"
#include "basis/AffineFunction.h"
#include "features/DenseFeatures.h"
#include "RandomGenerator.h"
#include "FileManager.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <random>
#include <cmath>
#include "MultiHeat.h"

using namespace std;
using namespace ReLe;
using namespace arma;


struct bboConfig
{
    unsigned int nbRuns, nbPolicies;
    double stepLength;
    StepRule* steprule = nullptr;

    virtual ~bboConfig()
    {
        if (steprule != nullptr)
            delete steprule;
    }
};

void help()
{
    cout << "mh_BBO algorithm [] #Updates #Policies stepLength [updaterule]" << endl;
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
            config.steprule = new ConstantStep(step_length);
        }
        else if (strcmp(argv[5+offset], "adaptive") == 0)
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
        config.nbRuns      = 200;
        config.nbPolicies  = 300;
        config.stepLength  = 0.11;
        config.steprule = new AdaptiveStep(config.stepLength);
    }
    //---

    FileManager fm("mh", "BBO");
    fm.createDir();
    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);

    MultiHeat mdp;
    vector<FiniteAction> actions;
    for (int i = 0; i < mdp.getSettings().finiteActionDim; ++i)
        actions.push_back(FiniteAction(i));

    //--- policy setup
    //input = [modo, t1, t2, azione]
    mat affineMtx(3,4);
    affineMtx << 0 << 1 << 0 << 0 << arma::endr
              << 0 << 0 << 1 << 0 << arma::endr
              << 1 << 0 << 0 << 0 << arma::endr;
    // crea redial basis function a griglia per t1 e t2
    BasisFunctions basis = GaussianRbf::generate({5,5}, {15,22,15,22});
    for (unsigned int i = 0; i < basis.size(); ++i)
    {
        GaussianRbf* rbf = (GaussianRbf*) basis[i];
        cout << rbf->getCenter().t() << " , " << rbf->getWidth().t() << endl;
    }
    // replica le RBFs per ogni modo
//    BasisFunctions repbasis = AndConditionBasisFunction::generate(basis,mdp.getSettings().continuosStateDim-1,mdp.getSettings().continuosStateDim);
    // transformate input for the replicated basis
    BasisFunctions abasis = AffineFunction::generate(basis, affineMtx);
    // replicate basis for each action
    BasisFunctions bfs = AndConditionBasisFunction::generate(abasis,mdp.getSettings().continuosStateDim,actions.size()-1);
    // create feature vector
    DenseFeatures phi(bfs);

//    // testare basis
//    vec inputtest = {0, 14, 21, 1};
//    vec outputtest = phi(inputtest);

//    cout << outputtest.t();

//    return 0;

    ParametricGibbsPolicy<DenseState> policy(actions, phi, 1.0/1e8);
    //---

    //--- distribution setup
    int nparams = phi.rows();
    DifferentiableDistribution* dist;

    if (strcmp(polType, "gauss") == 0)
    {
        //----- ParametricNormal
        arma::vec mean(nparams, fill::zeros);
        arma::mat cov(nparams, nparams, arma::fill::eye);
        dist = new ParametricNormal(mean, cov);
    }
    else if (strcmp(polType, "log") == 0)
    {
        //----- ParametricLogisticNormal
        dist = new ParametricLogisticNormal(nparams, 1);
    }
    else if (strcmp(polType, "chol") == 0)
    {
        //----- ParametricCholeskyNormal
        arma::vec mean(nparams, fill::zeros);
        arma::mat cov(nparams, nparams, arma::fill::eye);
        mat cholMtx = chol(cov);
        dist = new ParametricCholeskyNormal(mean, cholMtx);
    }
    else if (strcmp(polType, "diag") == 0)
    {
        //----- ParametricDiagonalNormal
        vec mean(nparams, fill::zeros);
        vec sigmas(nparams, fill::ones);
        dist = new ParametricDiagonalNormal(mean, sigmas);
        //-----
    }
    //---



    cout << "## MetaDistribution: " << dist->getDistributionName() << endl;

    int nbepperpol = 1, nbpolperupd = config.nbPolicies;
    char outputname[100];
    ReLe::Core<FiniteAction, DenseState>* core;
    if (strcmp(alg, "pgpe") == 0)
    {
        bool usebaseline = true;
        PGPE<FiniteAction, DenseState>* agent = new PGPE<FiniteAction, DenseState>
        (*dist, policy, nbepperpol, nbpolperupd, *(config.steprule), usebaseline);
        core = new ReLe::Core<FiniteAction, DenseState>(mdp, *agent);
        sprintf(outputname, "deep_pgpe_%s.log", polType);
    }
    else if (strcmp(alg, "nes") == 0)
    {
        bool usebaseline = true;
        NES<FiniteAction, DenseState>* agent = new NES<FiniteAction, DenseState>
        (*dist, policy, nbepperpol, nbpolperupd, *(config.steprule), usebaseline);
        core = new ReLe::Core<FiniteAction, DenseState>(mdp, *agent);
        sprintf(outputname, "deep_nes_%s.log", polType);
    }
    else if (strcmp(alg, "enes") == 0)
    {
        bool usebaseline = true;
        arma::vec mean(nparams, fill::zeros);
        arma::mat cov(nparams, nparams, arma::fill::eye);
        mat cholMtx = chol(cov);
        ParametricCholeskyNormal* distr = new ParametricCholeskyNormal(mean, cholMtx);
        eNES<FiniteAction, DenseState>* agent= new eNES<FiniteAction, DenseState>
        (*distr, policy, nbepperpol, nbpolperupd, *(config.steprule), usebaseline);
        core = new ReLe::Core<FiniteAction, DenseState>(mdp, *agent);
        sprintf(outputname, "deep_enes.log");
    }
    else if (strcmp(alg, "reps") == 0)
    {
        arma::vec mean(nparams, fill::zeros);
        arma::mat cov(nparams, nparams, arma::fill::eye);
        ParametricNormal* distr= new ParametricFullNormal(mean, cov);
        REPS<FiniteAction, DenseState>* agent = new REPS<FiniteAction, DenseState>(*distr,policy,nbepperpol,nbpolperupd);
        agent->setEps(0.9);
        core = new ReLe::Core<FiniteAction, DenseState>(mdp, *agent);
        sprintf(outputname, "deep_reps.log");
    }
    else
    {
        std::cout << "ERROR: Algorithm " << alg << " not found in (pgpe, nes, enes, reps)\n";
        abort();
    }

    WriteStrategy<FiniteAction, DenseState> wStrategy(
        fm.addPath(outputname),
        WriteStrategy<FiniteAction, DenseState>::AGENT,
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
            if (updateCount >= nbUpdates*every)
            {
                int p = 100 * updateCount/static_cast<double>(nbUpdates);
                cout << "### " << p << "% ###" << endl;
                core->getSettings().testEpisodeN = 100;
                arma::vec J = core->runBatchTest();
                cout << "mean score: " << J(0) << endl;
                every += bevery;
            }
        }
    }


    int nbTestEpisodes = 1000;
    cout << "Final test [#episodes: " << nbTestEpisodes << " ]" << endl;
    core->getSettings().testEpisodeN = nbTestEpisodes;
    cout << core->runBatchTest() << endl;

    //--- collect some trajectories
    core->getSettings().loggerStrategy = new WriteStrategy<FiniteAction, DenseState>(
        fm.addPath("mhFinal.log"),
        WriteStrategy<FiniteAction, DenseState>::TRANS,
        true /*delete file*/
    );
    for (int n = 0; n < 100; ++n)
        core->runTestEpisode();
    //---

    delete dist;
    delete core;
    return 0;
}
