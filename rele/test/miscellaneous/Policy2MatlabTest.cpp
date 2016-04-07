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

#include "rele/policy/Policy.h"
#include "rele/policy/parametric/differentiable/NormalPolicy.h"
#include "rele/policy/parametric/differentiable/LinearPolicy.h"
#include "rele/policy/parametric/differentiable/PortfolioNormalPolicy.h"
#include "rele/policy/parametric/differentiable/GenericGibbsPolicy.h"
#include "rele/policy/parametric/differentiable/GenericNormalPolicy.h"
#include "rele/policy/parametric/differentiable/ParametricMixturePolicy.h"
#include "rele/policy/nonparametric/RandomPolicy.h"
#include "rele/utils/RandomGenerator.h"
#include "rele/utils/FileManager.h"
#include "rele/approximators/basis/IdentityBasis.h"
#include "rele/approximators/basis/PolynomialFunction.h"
#include "rele/approximators/features/DenseFeatures.h"
#include "rele/approximators/features/SparseFeatures.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <cmath>

using namespace std;
using namespace ReLe;
using namespace arma;

//TODO [CLEANUP] implement or remove test

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

        BasisFunctions basis = PolynomialFunction::generate(deg(0),state.n_elem);
        DenseFeatures phi(basis);

        //----- NormalPolicy
        NormalPolicy policy(stddev(0), phi);
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
    else if (strcmp(argv[1], "normalstdstate") == 0)
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

        //load degree mean
        arma::vec deg;
        deg.load(argv[5], raw_ascii);

        //define basis for mean
        BasisFunctions basis = PolynomialFunction::generate(deg(0), state.n_elem);
        DenseFeatures phi(basis);

        //load degree stddev
        arma::vec degs;
        degs.load(argv[6], raw_ascii);

        //load weights stddev
        arma::vec initws;
        initws.load(argv[7], raw_ascii);

        //define basis for stddev
        BasisFunctions basiss = PolynomialFunction::generate(degs(0), state.n_elem);
        assert(initws.n_elem == basiss.size());
        DenseFeatures phis(basiss);

        //----- NormalStateDependantStddevPolicy
        NormalStateDependantStddevPolicy policy(phi, phis, initws);
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
    else if (strcmp(argv[1], "mvnstdstate") == 0)
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

        //load degree mean
        arma::vec deg;
        deg.load(argv[5], raw_ascii);

        //define basis for mean
        BasisFunctions basis = PolynomialFunction::generate(deg(0), state.n_elem);
        Features* phi = nullptr;

        if (action.n_elem == 1)
        {
            phi = new DenseFeatures(basis);
        }
        else
        {
            phi = new SparseFeatures(basis, action.n_elem);
        }


        //load degree stddev
        arma::vec degs;
        degs.load(argv[6], raw_ascii);

        //load weights stddev
        arma::vec initws;
        initws.load(argv[7], raw_ascii);

        //define basis for stddev
        BasisFunctions basiss = PolynomialFunction::generate(degs(0), state.n_elem);
        Features* phis = nullptr;

        if (action.n_elem == 1)
        {
            phis = new DenseFeatures(basiss);
        }
        else
        {
            phis = new SparseFeatures(basiss, action.n_elem);
        }

        cerr << phis->operator ()(state) << endl;

        assert(initws.n_elem == phis->rows());
        LinearApproximator regm(*phi);
        LinearApproximator regs(*phis);

        //----- NormalStateDependantStddevPolicy
        GenericMVNStateDependantStddevPolicy policy(regm, regs);
        cerr << vectorize(params,initws) << endl;
        cerr << policy.getParametersSize() << endl;
        policy.setParameters(vectorize(params,initws));

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
    else if (strcmp(argv[1], "paramgibbs") == 0)
    {
        //load policy parameters
        vec params;
        params.load(argv[2], raw_ascii);

        //load inverse temperature
        vec inverseT;
        inverseT.load(argv[3], raw_ascii);

        //load naction
        vec nactions;
        nactions.load(argv[4], raw_ascii);

        //load state
        vec state;
        state.load(argv[5], raw_ascii);

        //load action
        vec action;
        action.load(argv[6], raw_ascii);

        //load degree
        arma::vec deg;
        deg.load(argv[7], raw_ascii);

        BasisFunctions basis = PolynomialFunction::generate(deg(0),1+state.n_elem);
        for (int i = 0; i < basis.size(); ++i)
            std::cout << *(basis[i]) << std::endl;
        DenseFeatures phi(basis);
        LinearApproximator reg(phi);

        //----- NormalPolicy
        vector<FiniteAction> actions;
        for (int i = 0; i < nactions(0); ++i)
            actions.push_back(FiniteAction(i));

        GenericParametricGibbsPolicy<DenseState> policy(actions, reg, 1.0/inverseT(0));
        policy.setParameters(params);

        cout << policy.getPolicyName() << endl;
        int dim = action.n_elem, nbs = 50000;

        //draw random points
        mat P(nbs,dim);
        for (int i = 0; i < nbs; ++i)
        {
            unsigned int sample = policy(state);
            P(i,0) = sample;
        }
        P.save(fm.addPath("samples.dat"), raw_ascii);

        //evaluate density
        vec density(1);
        density(0) = policy(state,action(0));
        density.save(fm.addPath("density.dat"), raw_ascii);

        //compute gradient
        vec grad = policy.difflog(state,action(0));
        grad.save(fm.addPath("grad.dat"), raw_ascii);

//        //compute hessian
//        mat hess = policy.diff2log(state,action(0));
//        hess.save(fm.addPath("hessian.dat"), raw_ascii);

    }
    else if (strcmp(argv[1], "mvn") == 0)
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

        //as variance
        arma::mat variance;
        variance.load(argv[6], raw_ascii);

        //define approximation
        BasisFunctions basis = PolynomialFunction::generate(deg(0), state.n_elem);
        Features* phi = nullptr;

        if (action.n_elem == 1)
        {
            phi = new DenseFeatures(basis);
        }
        else
        {
            phi = new SparseFeatures(basis, action.n_elem);
        }

        //define policy
        cerr << params.t();
        MVNPolicy policy(*phi, variance);
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

        if (phi != nullptr)
            delete phi;
    }
    else if (strcmp(argv[1], "genericmvn") == 0)
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

        //as variance
        arma::mat variance;
        variance.load(argv[6], raw_ascii);

        //define approximation
        BasisFunctions basis = PolynomialFunction::generate(deg(0), state.n_elem);
        Features* phi = nullptr;

        if (action.n_elem == 1)
        {
            phi = new DenseFeatures(basis);
        }
        else
        {
            phi = new SparseFeatures(basis, action.n_elem);
        }

        //define policy
        cerr << params.t();
        LinearApproximator reg(*phi);
        GenericMVNPolicy policy(reg, variance);
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

        if (phi != nullptr)
            delete phi;
    }
    else if (strcmp(argv[1], "mvnlog") == 0)
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

        //as variance
        arma::vec as_variance;
        as_variance.load(argv[6], raw_ascii);

        //define approximation
        BasisFunctions basis = PolynomialFunction::generate(deg(0), state.n_elem);
        Features* phi = nullptr;

        if (action.n_elem == 1)
        {
            phi = new DenseFeatures(basis);
        }
        else
        {
            phi = new SparseFeatures(basis, action.n_elem);
        }

        //define policy
        cerr << params.t();
        MVNLogisticPolicy policy(*phi, as_variance);
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

        if (phi != nullptr)
            delete phi;

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
        BasisFunctions basis = PolynomialFunction::generate(deg(0), state.n_elem);
        Features* phi = nullptr;

        if (action.n_elem == 1)
        {
            phi = new DenseFeatures(basis);
        }
        else
        {
            phi = new SparseFeatures(basis, action.n_elem);
        }

        //define policy
        cerr << params.t();
        MVNDiagonalPolicy policy(*phi);
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

        if (phi != nullptr)
            delete phi;

    }
    else if (strcmp(argv[1], "genericmvndiag") == 0)
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
        BasisFunctions basis = PolynomialFunction::generate(deg(0), state.n_elem);
        Features* phi = nullptr;

        if (action.n_elem == 1)
        {
            phi = new DenseFeatures(basis);
        }
        else
        {
            phi = new SparseFeatures(basis, action.n_elem);
        }

        //define policy
        cerr << params.t();
        LinearApproximator reg(*phi);
        GenericMVNDiagonalPolicy policy(reg);
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

        if (phi != nullptr)
            delete phi;

    }
    else if (strcmp(argv[1], "portfolio") == 0)
    {
        //load policy parameters
        vec params;
        params.load(argv[2], raw_ascii);

        //load epsilon
        vec epsilon;
        epsilon.load(argv[3], raw_ascii);

        //load state
        vec state;
        state.load(argv[4], raw_ascii);
        assert(state.n_elem == 6);

        //load action
        vec action;
        action.load(argv[5], raw_ascii);

        BasisFunctions basis = IdentityBasis::generate(6);
        DenseFeatures phi(basis);

        //----- PortfolioNormalPolicy
        PortfolioNormalPolicy policy(epsilon(0), phi);
        policy.setParameters(params);


        cout << policy.getPolicyName() << endl;
        int dim = action.n_elem, nbs = 50000;


        //draw random points
        mat P(nbs,dim);
        for (int i = 0; i < nbs; ++i)
        {
            unsigned int u = policy(state);
            P(i,0) = u;
        }
        P.save(fm.addPath("samples.dat"), raw_ascii);


        //evaluate density
        vec density(1);
        density(0) = policy(state,action(0));
        density.save(fm.addPath("density.dat"), raw_ascii);

        //compute gradient
        vec grad = policy.difflog(state,action(0));
        grad.save(fm.addPath("grad.dat"), raw_ascii);

        //compute hessian
        mat hess = policy.diff2log(state,action(0));
        hess.save(fm.addPath("hessian.dat"), raw_ascii);
    }
    else if (strcmp(argv[1], "genericmix") == 0)
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

        //as variance
        arma::mat variance;
        variance.load(argv[6], raw_ascii);

        //nbComponents
        arma::mat nbComp;
        nbComp.load(argv[7], raw_ascii);

        //define approximation
        BasisFunctions basis = PolynomialFunction::generate(deg(0), state.n_elem);
        Features* phi = nullptr;

        if (action.n_elem == 1)
        {
            phi = new DenseFeatures(basis);
        }
        else
        {
            phi = new SparseFeatures(basis, action.n_elem);
        }

        //define policy
        cerr << params.t();
        std::vector<DifferentiablePolicy<DenseAction,DenseState>*> vp;
        for (int i = 0;  i < nbComp(0,0); ++i)
        {
            LinearApproximator* reg = new LinearApproximator(*phi);
            vp.push_back(new GenericMVNPolicy(*reg, variance));
        }

        GenericParametricMixturePolicy<DenseAction,DenseState> policy(vp);

        cerr << "nbParams: " << policy.getParametersSize() << endl;

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

        if (phi != nullptr)
            delete phi;

    }
    else if (strcmp(argv[1], "genericlogmix") == 0)
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

        //as variance
        arma::mat variance;
        variance.load(argv[6], raw_ascii);

        //nbComponents
        arma::mat nbComp;
        nbComp.load(argv[7], raw_ascii);

        //define approximation
        BasisFunctions basis = PolynomialFunction::generate(deg(0), state.n_elem);
        Features* phi = nullptr;

        if (action.n_elem == 1)
        {
            phi = new DenseFeatures(basis);
        }
        else
        {
            phi = new SparseFeatures(basis, action.n_elem);
        }

        //define policy
        cerr << params.t();
        std::vector<DifferentiablePolicy<DenseAction,DenseState>*> vp;
        for (int i = 0;  i < nbComp(0,0); ++i)
        {
            LinearApproximator* reg = new LinearApproximator(*phi);
            vp.push_back(new GenericMVNPolicy(*reg, variance));
        }

        GenericParametricLogisticMixturePolicy<DenseAction,DenseState> policy(vp);

        cerr << "nbParams: " << policy.getParametersSize() << endl;

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

        if (phi != nullptr)
            delete phi;

    }
    else if (strcmp(argv[1], "linear") == 0)
    {
        //----- DetLinearPolicy
        abort();
    }
    else if (strcmp(argv[1], "random") == 0)
    {
        //----- RandomPolicy
        abort();
    }
    else if (strcmp(argv[1], "randomdiscrete") == 0)
    {
        //----- StochasticDiscretePolicy
        abort();
    }
    else if (strcmp(argv[1], "randomdiscretebias") == 0)
    {
        //----- RandomDiscreteBiasPolicy
        abort();
    }
    else
    {
        cerr << "Unknown policy!" << endl;
        abort();
    }

    return 0;
}
