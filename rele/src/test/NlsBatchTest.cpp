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

#include "DifferentiableNormals.h"
#include "Core.h"
#include "PolicyEvalAgent.h"
#include "parametric/differentiable/NormalPolicy.h"
#include "policy_search/offpolicy/OffPolicyREINFORCE.h"
#include "policy_search/offpolicy/OffPolicyGPOMDP.h"
#include "BasisFunctions.h"
#include "basis/PolynomialFunction.h"
#include "RandomGenerator.h"
#include "FileManager.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <random>
#include <cmath>
#include "NLS.h"

using namespace std;
using namespace ReLe;
using namespace arma;

int main(int argc, char *argv[])
{
    FileManager fm("Offpolicy", "nls");
    fm.createDir();
    fm.cleanDir();

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
    int dim = mdp.getSettings().continuosStateDim;

    //--- define policy (low level)
    DenseBasisVector basis;
    basis.generatePolynomialBasisFunctions(1,dim);
    delete basis.at(0);
    basis.erase(basis.begin());
    cout << "--- Mean regressor ---" << endl;
    cout << basis << endl;
    LinearApproximator behave_meanRegressor(dim, basis);
    arma::vec wB(2);
    wB(0) = -0.57;
    wB(1) = 0.5;
    behave_meanRegressor.setParameters(wB);

    LinearApproximator target_meanRegressor(dim, basis);

    DenseBasisVector stdBasis;
    stdBasis.generatePolynomialBasisFunctions(1,dim);
    delete stdBasis.at(0);
    stdBasis.erase(stdBasis.begin());
    cout << "--- Standard deviation regressor ---" << endl;
    cout << stdBasis << endl;
    LinearApproximator stdRegressor(dim, stdBasis);
    arma::vec stdWeights(stdRegressor.getParametersSize());
    stdWeights.fill(0.5);
    stdRegressor.setParameters(stdWeights);


    NormalStateDependantStddevPolicy behavioral(&behave_meanRegressor, &stdRegressor);
    NormalStateDependantStddevPolicy target(&target_meanRegressor, &stdRegressor);
    //---

    PolicyEvalAgent<DenseAction,DenseState> agent(behavioral);

    ReLe::Core<DenseAction, DenseState> oncore(mdp, agent);
    CollectorStrategy<DenseAction, DenseState>* strat = new CollectorStrategy<DenseAction, DenseState>();
    oncore.getSettings().loggerStrategy = strat;

    int horiz = mdp.getSettings().horizon;
    oncore.getSettings().episodeLenght = horiz;

    int nbTrajectories = 20e3;
    for (int n = 0; n < nbTrajectories; ++n)
        oncore.runTestEpisode();

    Dataset<DenseAction, DenseState>& data = strat->data;
    ofstream out(fm.addPath("Dataset.csv"), ios_base::out);
    if (out.is_open())
        data.writeToStream(out);
    out.close();

    cout << "# Ended data collection" << endl;


    AdaptiveStep stepl(0.1);
//    OffpolicyREINFORCE<DenseAction, DenseState> offagent(target, behavioral, data.size(), stepl);
    OffPolicyGPOMDP<DenseAction, DenseState> offagent(target, behavioral, data.size(), 0.1*data.size(), horiz, stepl);
    BatchCore<DenseAction, DenseState> offcore(mdp, offagent, data);
    offcore.getSettings().loggerStrategy = new WriteStrategy<DenseAction, DenseState>(
        fm.addPath("nls.log"),
        WriteStrategy<DenseAction, DenseState>::AGENT,
        true /*delete file*/
    );
    offcore.getSettings().episodeLenght = horiz;

    int nbUpdates = 40;
    double every, bevery;
    every = bevery = 0.1; //%


    for (int i = 0; i < nbUpdates; i++)
    {
        offcore.processBatchData();
        int p = 100 * i/static_cast<double>(nbUpdates);
        cout << "### " << p << "% ###" << endl;
        //                cout << dist.getParameters().t();
        offcore.getSettings().testEpisodeN = 100;
        arma::vec J = offcore.runBatchTest();
        cout << "mean score: " << J(0) << endl;
        every += bevery;
    }

    delete strat;
    return 0;
}
