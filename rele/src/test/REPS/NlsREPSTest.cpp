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

#include "policy_search/REPS/REPS.h"
#include "DifferentiableNormals.h"
#include "Core.h"
#include "parametric/differentiable/NormalPolicy.h"
#include "BasisFunctions.h"
#include "basis/IdentityBasis.h"
#include "features/DenseFeatures.h"

#include "RandomGenerator.h"

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
    /*FileManager fm("NLS", "REPS");
    fm.createDir();
    fm.cleanDir();*/

    NLS mdp;
    //with these settings
    //max in ( -3.58, 10.5 ) -> J = 8.32093
    //note that there are multiple optimal solutions
    //TODO: verificare, serve interfaccia core per valutare una politica

    int dim = mdp.getSettings().continuosStateDim;
    cout << "dim: " << dim << endl;


    //--- define meta distribution (high-level policy)
    arma::vec mean(dim, arma::fill::zeros);
    mean[0] = -0.42;
    mean[1] =  0.42;
    arma::mat cov(dim, dim, arma::fill::eye);
    cov *= 0.1;
    ParametricNormal dist(mean,cov);
    //---


    //--- define policy (low level)
    BasisFunctions basis = IdentityBasis::generate(dim);
    DenseFeatures phi(basis);

    BasisFunctions stdBasis = IdentityBasis::generate(dim);
    DenseFeatures stdPhi(stdBasis);
    arma::vec stdWeights(stdPhi.rows());
    stdWeights.fill(0.5);


    NormalStateDependantStddevPolicy policy(phi, stdPhi, stdWeights);
    //---

    int nbepperpol = 1, nbpolperupd = 300;
    REPS<DenseAction, DenseState, ParametricNormal> agent(dist,policy,nbepperpol,nbpolperupd);
    agent.setEps(0.5);

    ReLe::Core<DenseAction, DenseState> core(mdp, agent);

    int episodes  = 10000;
    for (int i = 0; i < episodes; i++)
    {
        core.getSettings().episodeLenght = mdp.getSettings().horizon;
        cout << "starting episode" << endl;
        core.runEpisode();
    }

    cout << dist.getMean().t() << endl;
    cout << dist.getCovariance() << endl;

    return 0;
}
