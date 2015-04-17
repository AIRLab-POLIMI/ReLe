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

#include "Policy.h"
#include "parametric/differentiable/NormalPolicy.h"
#include "parametric/differentiable/LinearPolicy.h"
#include "parametric/differentiable/PortfolioNormalPolicy.h"
#include "RandomGenerator.h"
#include "FileManager.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <cmath>

using namespace std;
using namespace ReLe;
using namespace arma;

void help(char* argv[])
{
    cout << "### Policy Test ###" << endl;
}

int main(int argc, char *argv[])
{

    if ((argc == 0) || (argc < 5))
    {
        cout << argc << endl;
        help(argv);
        return 0;
    }


    FileManager fm("pol2mat", "test");
    fm.createDir();
    fm.cleanDir();


    if (strcmp(argv[1], "normal") == 0)
    {
        //load policy parameters
        vec params;
        params.load(argv[2], raw_ascii);

        //load stddeviation
        vec stddev;
        stddev.load(argv[3], raw_ascii);

        //load state
        vec state;
        state.load(argv[4], raw_ascii);

        //load action
        vec action;
        action.load(argv[5], raw_ascii);

        //load degree
        arma::vec deg;
        deg.load(argv[6], raw_ascii);

        DenseBasisVector basis;
        basis.generatePolynomialBasisFunctions(deg(0),state.n_elem);
        LinearApproximator la(state.n_elem, basis);

        //----- NormalPolicy
        NormalPolicy policy(stddev(0), &la);
        policy.setParameters(params);

        cout << policy.getPolicyName() << endl;
        int dim = action.n_elem, nbs = 50000;

        //draw random points
        mat P(nbs,dim);
        for (int i = 0; i < nbs; ++i)
        {
            arma::vec sample = policy(state);
            for (int k = 0; k < dim; ++k)
                P(i,k) = sample[k];
        }
        P.save(fm.addPath("samples.dat"), raw_ascii);

        //evaluate density
        vec density(1);
        density(0) = policy(state,action);
        density.save(fm.addPath("density.dat"), raw_ascii);

        //compute gradient
        vec grad = policy.difflog(state,action);
        grad.save(fm.addPath("grad.dat"), raw_ascii);

        //compute hessian
        mat hess = policy.diff2log(state,action);
        hess.save(fm.addPath("hessian.dat"), raw_ascii);

    }
    else if (strcmp(argv[1], "normalstd") == 0)
    {
        //TODO update
        arma::vec point;
        point.load(argv[2], raw_ascii);

        arma::vec deg;
        deg.load(argv[3], raw_ascii);

        arma::vec initw;
        initw.load(argv[4], raw_ascii);

        DenseBasisVector basis;
        basis.generatePolynomialBasisFunctions(deg(0),point.n_elem);

        assert(initw.n_elem == basis.size());

        LinearApproximator lam(point.n_elem, basis);
        lam.setParameters(initw);

        arma::vec degs;
        degs.load(argv[5], raw_ascii);

        arma::vec initws;
        initws.load(argv[6], raw_ascii);

        DenseBasisVector basiss;
        basiss.generatePolynomialBasisFunctions(degs(0),point.n_elem);

        assert(initws.n_elem == basiss.size());

        LinearApproximator las(point.n_elem, basiss);
        las.setParameters(initws);

        //----- NormalStateDependantStddevPolicy
        vec varas;
        varas.load(argv[4], raw_ascii);
        NormalStateDependantStddevPolicy* policy = new NormalStateDependantStddevPolicy(&lam, &las);
    }
    else if (strcmp(argv[1], "mvn") == 0)
    {
        //TODO!!
        abort();

        //----- MVNPolicy
        //        policy = new MVNPolicy(p1, p2);
        //        point.load(argv[4], raw_ascii);
    }
    else if (strcmp(argv[1], "mvndiag") == 0)
    {
        //load policy parameters
        vec params;
        params.load(argv[2], raw_ascii);

        //load state
        vec state;
        state.load(argv[3], raw_ascii);

        //load action
        vec action;
        action.load(argv[4], raw_ascii);

        //load degree
        arma::vec deg;
        deg.load(argv[5], raw_ascii);

        //define approximation
        DenseBasisVector basis;
        basis.generatePolynomialBasisFunctions(deg(0), state.n_elem);
        cout << basis;
        LinearApproximator lam(state.n_elem, basis);

        //define policy
        cerr << params.t();
        MVNDiagonalPolicy policy(&lam);
        policy.setParameters(params);


        cout << policy.getPolicyName() << endl;
        int dim = action.n_elem, nbs = 50000;


        //draw random points
        mat P(nbs,dim);
        for (int i = 0; i < nbs; ++i)
        {
            arma::vec sample = policy(state);
            for (int k = 0; k < dim; ++k)
                P(i,k) = sample[k];
        }
        P.save(fm.addPath("samples.dat"), raw_ascii);


        //evaluate density
        vec density(1);
        density(0) = policy(state,action);
        density.save(fm.addPath("density.dat"), raw_ascii);

        //compute gradient
        vec grad = policy.difflog(state,action);
        grad.save(fm.addPath("grad.dat"), raw_ascii);

        //compute hessian
        mat hess = policy.diff2log(state,action);
        hess.save(fm.addPath("hessian.dat"), raw_ascii);

    }
    else if (strcmp(argv[1], "portfolio") == 0)
    {
        //TODO update
        arma::vec point;

        arma::vec epsilon;
        epsilon.load(argv[2], raw_ascii);

        point.load(argv[3], raw_ascii);

        arma::vec deg;
        deg.load(argv[4], raw_ascii);

        arma::vec initw;
        initw.load(argv[5], raw_ascii);

        DenseBasisVector basis;
        basis.generatePolynomialBasisFunctions(deg(0),point.n_elem);

        assert(initw.n_elem == basis.size());

        LinearApproximator la(point.n_elem, basis);
        la.setParameters(initw);

        //----- PortfolioNormalPolicy
        PortfolioNormalPolicy* policy = new PortfolioNormalPolicy(epsilon(0), &la);



        arma::vec action;
        action.load(argv[6], raw_ascii);

        cout << policy->getPolicyName() << endl;


        int dim = point.n_elem, nbs = 50000;

        //draw random points
        mat P(nbs,1);
        for (int i = 0; i < nbs; ++i)
        {
            unsigned int sample = (*policy)(point);
            P(i,0) = sample;
        }
        P.save("/tmp/pol2matlab/samples.dat", raw_ascii);

        //compute gradient
        vec grad = policy->difflog(point,action(0));
        grad.save("/tmp/pol2matlab/grad.dat", raw_ascii);

        //compute hessian
        mat hess = policy->diff2log(point,action(0));
        hess.save("/tmp/pol2matlab/hess.dat", raw_ascii);
    }
    else if (strcmp(argv[1], "linear") == 0)
    {
        //----- DetLinearPolicy
        //TODO!!
        abort();
    }
    else
    {
        cerr << "Unknown policy!" << endl;
        abort();
    }

    return 0;
}
