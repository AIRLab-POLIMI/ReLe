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

#include "Rocky.h"
#include "policy_search/REPS/REPS.h"
#include "DifferentiableNormals.h"
#include "Core.h"
#include "parametric/differentiable/LinearPolicy.h"
#include "BasisFunctions.h"

#include "FileManager.h"
#include "ConsoleManager.h"

#include <iostream>

using namespace std;
using namespace ReLe;
using namespace arma;



namespace ReLe
{
class RockyCircularPolicy : public ParametricPolicy<DenseAction, DenseState>
{
public:

    RockyCircularPolicy() : v(1)
    {

    }

    //Policy
    virtual vec operator() (const vec& state)
    {
        vec u(3);
        u[0] = 1;
        u[1] = M_PI/16;
        u[2] = 0;

        return u;
    }
    virtual double operator() (const vec& state, const vec& action)
    {
        return 0;
    }

    virtual std::string getPolicyName()
    {
        return "Circular fake";
    }

    virtual std::string getPolicyHyperparameters()
    {
        return "";
    }

    virtual std::string printPolicy()
    {
        return "";
    }

    //ParametricPolicy
    virtual const arma::vec& getParameters() const
    {
        return v;
    }

    virtual const unsigned int getParametersSize() const
    {
        return 1;
    }

    virtual void setParameters(arma::vec& w)
    {

    }


private:
    vec v;
};

}

int main(int argc, char *argv[])
{
	FileManager fm("Rocky", "REPS");
	fm.createDir();
	fm.cleanDir();

    Rocky rocky;

    //-- Low level policy
    int dim = rocky.getSettings().continuosStateDim;
    int actionDim = rocky.getSettings().continuosActionDim;
    DenseBasisVector basis;
    basis.generatePolynomialBasisFunctions(1, dim);
    SparseBasisMatrix basisMatrix(basis, actionDim);

    LinearApproximator regressor(dim, basisMatrix);

    DetLinearPolicy<DenseState> policy(&regressor);
    //RockyCircularPolicy policy;
    //--

    //-- high level policy
    int dimPolicy = policy.getParametersSize();
    arma::vec mean(dimPolicy, fill::zeros);
    arma::mat cov(dimPolicy, dimPolicy, fill::eye);

    cov *= 10;


    ParametricNormal dist(mean, cov);
    //--

    //-- REPS agent
    int nbepperpol = 1, nbpolperupd = 50;
    REPS<DenseAction, DenseState, ParametricNormal> agent(dist,policy,nbepperpol,nbpolperupd);
    agent.setEps(0.5);

    Core<DenseAction, DenseState> core(rocky, agent);
    //--


    int episodes = nbepperpol*nbpolperupd*60;
    core.getSettings().episodeLenght = 10000;
    core.getSettings().loggerStrategy = new WriteStrategy<DenseAction, DenseState>(fm.addPath("Rocky.log"));


    ConsoleManager console(episodes, 1);
    console.printInfo("starting learning");
    for (int i = 0; i < episodes; i++)
    {
        console.printProgress(i);
        core.runEpisode();
    }

    console.printInfo("Starting evaluation episode");
    core.runTestEpisode();

    delete core.getSettings().loggerStrategy;

    cout << dist.getMean().t() << endl;
    cout << dist.getCovariance() << endl;

    return 0;

}
