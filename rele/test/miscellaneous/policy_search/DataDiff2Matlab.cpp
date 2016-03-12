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
#include "rele/policy/parametric/differentiable/GenericGibbsPolicy.h"
#include "rele/approximators/BasisFunctions.h"
#include "rele/policy/parametric/differentiable/NormalPolicy.h"
#include "rele/approximators/basis/PolynomialFunction.h"
#include "rele/approximators/basis/GaussianRbf.h"
#include "rele/approximators/basis/ConditionBasedFunction.h"
#include "rele/approximators/features/DenseFeatures.h"

#include "rele/environments/Dam.h"
#include "rele/core/PolicyEvalAgent.h"
#include "rele/algorithms/policy_search/PGPE/PGPE.h"
#include "rele/IRL/ParametricRewardMDP.h"
#include "rele/utils/RandomGenerator.h"
#include "rele/utils/FileManager.h"

#include "rele/algorithms/policy_search/gradient/FunctionGradient.h"
#include "rele/algorithms/policy_search/gradient/FunctionHessian.h"
#include "rele/algorithms/policy_search/gradient/PolicyGradientAlgorithm.h"

using namespace std;
using namespace ReLe;
using namespace arma;

void help()
{
    cout << "datadiff2mat [algorithm] parameters_file" << endl;
    cout << " - algorithm: r, rb, g, gb (default)" << endl;
}


int main(int argc, char *argv[])
{
//    RandomGenerator::seed(418932850);

    /*** check inputs ***/
    vec params;
    double gamma;
    if (argc == 4)
    {
        if (strcmp(argv[1], "r") == 0)
        {
            cout << "REINFORCE" << endl;
        }
        else if (strcmp(argv[1], "rb") == 0)
        {
            cout << "REINFORCE BASE" << endl;
        }
        else if (strcmp(argv[1], "g") == 0)
        {
            cout << "GPOMDP" << endl;
        }
        else if (strcmp(argv[1], "gb") == 0)
        {
            cout << "GPOMDP BASE" << endl;
        }
        else if (strcmp(argv[1], "enac") == 0)
        {
            cout << "ENAC" << endl;
        }
        else if (strcmp(argv[1], "enacb") == 0)
        {
            cout << "ENAC BASE" << endl;
        }
        else if ((strcmp(argv[1], "natr") == 0) || (strcmp(argv[1], "natrb") == 0) ||
                 (strcmp(argv[1], "natg") == 0) || (strcmp(argv[1], "natgb") == 0) )
        {
            cout << "NATURAL GRADIENT" << endl;
        }
        else
        {
            std::cout << "Error unknown argument " << argv[1] << std::endl;
            help();
            exit(1);
        }

        params.load(argv[2], raw_ascii);
        gamma = atof(argv[3]);
    }
    else
    {
        help();
        return 1;
    }
    /******/

    FileManager fm("datadiff2mat", "test");
    fm.createDir();
    //    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);

    /*** Set up MDP ***/
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
    NormalPolicy policy(0.1, phi);
    policy.setParameters(params);
    //---

    cout << policy.getParameters().t();


    PolicyEvalAgent<DenseAction, DenseState> expert(policy);

    /* Generate DAM expert dataset */
    Core<DenseAction, DenseState> expertCore(mdp, expert);
    CollectorStrategy<DenseAction, DenseState> collection;
    expertCore.getSettings().loggerStrategy = &collection;
    expertCore.getSettings().episodeLength = mdp.getSettings().horizon;
    expertCore.getSettings().testEpisodeN = 200;
    expertCore.runTestEpisodes();

    Dataset<DenseAction,DenseState>& data = collection.data;
    ofstream datasetfile(fm.addPath("dataset.dat"));
    if (datasetfile.is_open())
    {
        data.writeToStream(datasetfile);
        datasetfile.close();
    }


    vec gradient;
    mat hessian;
    GradientFromDataWorker<DenseAction, DenseState> gdw(data, policy, gamma, 0);
    HessianFromDataWorker<DenseAction, DenseState, NormalPolicy> hdw(data, policy, gamma, 0);
    if (strcmp(argv[1], "r") == 0)
    {
        cout << "PG REINFORCE" << endl;
        gradient = gdw.ReinforceGradient();
        hessian = hdw.ReinforceHessian();
    }
    else if (strcmp(argv[1], "rb") == 0)
    {
        cout << "PG REINFORCE BASE" << endl;
        gradient = gdw.ReinforceBaseGradient();
    }
    else if (strcmp(argv[1], "g") == 0)
    {
        cout << "PG GPOMDP" << endl;
        gradient = gdw.GpomdpGradient();
    }
    else if (strcmp(argv[1], "gb") == 0)
    {
        cout << "PG GPOMDP BASE" << endl;
        gradient = gdw.GpomdpBaseGradient();
    }
    else if (strcmp(argv[1], "enac") == 0)
    {
        cout << "PG ENAC" << endl;
        gradient = gdw.ENACGradient();
    }
    else if (strcmp(argv[1], "enacb") == 0)
    {
        cout << "PG ENAC BASE" << endl;
        gradient = gdw.ENACBaseGradient();
    }
    else if (strcmp(argv[1], "natr") == 0)
    {
        cout << "PG NAT R " << endl;
        gradient = gdw.NaturalGradient(GradientFromDataWorker<DenseAction, DenseState>::NaturalGradType::NATR);
    }
    else if (strcmp(argv[1], "natrb") == 0)
    {
        cout << "PG NAT RB BASE" << endl;
        gradient = gdw.NaturalGradient(GradientFromDataWorker<DenseAction, DenseState>::NaturalGradType::NATRB);
    }
    else if (strcmp(argv[1], "natg") == 0)
    {
        cout << "PG NAT G " << endl;
        gradient = gdw.NaturalGradient(GradientFromDataWorker<DenseAction, DenseState>::NaturalGradType::NATG);
    }
    else if (strcmp(argv[1], "natgb") == 0)
    {
        cout << "PG NAT GB BASE" << endl;
        gradient = gdw.NaturalGradient(GradientFromDataWorker<DenseAction, DenseState>::NaturalGradType::NATGB);
    }

    gradient.save(fm.addPath("gradient.dat"), raw_ascii);
    hessian.save(fm.addPath("hessian.dat"), raw_ascii);

    return 0;
}
