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

#ifndef GIRL_H_
#define GIRL_H_

#include "rele/IRL/IRLAlgorithm.h"
#include "rele/utils/ArmadilloExtensions.h"
//#include "rele/algorithms/policy_search/step_rules/StepRules.h"

#include "rele/IRL/utils/GradientCalculatorFactory.h"
#include "rele/IRL/utils/Optimization.h"

#include <nlopt.hpp>
#include <cassert>
#include <stdexcept>

namespace ReLe
{

template<class ActionC, class StateC>
class GIRL: public IRLAlgorithm<ActionC, StateC>
{
public:

    GIRL(Dataset<ActionC, StateC>& dataset,
         DifferentiablePolicy<ActionC, StateC>& policy,
         LinearApproximator& rewardf, double gamma, IrlGrad aType,
         bool useSimplexConstraints = true) :
        policy(policy), data(dataset), rewardf(rewardf), gamma(gamma), aType(aType)
    {
        nbFunEvals = 0;

        // build gradient calculator
        gradientCalculator = GradientCalculatorFactory<ActionC, StateC>::build(aType, rewardf.getFeatures(),
                             dataset, policy, gamma);

        // initially all features are active
        active_feat.set_size(rewardf.getParametersSize());
        std::iota(std::begin(active_feat), std::end(active_feat), 0);
    }

    virtual ~GIRL()
    {
    }

    //======================================================================
    // RUNNERS
    //----------------------------------------------------------------------
    virtual void run() override
    {
        run(arma::vec(), 0);
    }

    virtual void run(arma::vec starting, unsigned int maxFunEvals)
    {
        int dpr = rewardf.getParametersSize();
        assert(dpr > 0);

        if (maxFunEvals == 0)
            maxFunEvals = std::min(30 * dpr, 600);

        nbFunEvals = 0;

        // initialize active features set
        preprocessing();

        //compute effective parameters dimension
        int effective_dim = active_feat.n_elem;

        // handle the case of only one active reward feature
        if (effective_dim == 1)
        {
            arma::vec x(dpr, arma::fill::zeros);
            x.elem(active_feat).ones();
            rewardf.setParameters(x);
            return;
        }

        //setup optimization algorithm
        nlopt::opt optimizator;

        // simplex constraint reduces the parameter by one element
        --effective_dim;

        optimizator = nlopt::opt(nlopt::algorithm::LD_SLSQP, effective_dim);

        std::vector<double> lowerBounds(effective_dim, 0.0);
        std::vector<double> upperBounds(effective_dim, 1.0);
        optimizator.set_lower_bounds(lowerBounds);
        optimizator.set_upper_bounds(upperBounds);

        // equality constraint
        // optimizator.add_equality_constraint(GIRL::OneSumConstraint, NULL, 1e-3);

        // inequality constraint
        // x >= 0 && sum x <= 1
        std::vector<double> tols(effective_dim + 1, 1e-5);
        optimizator.add_inequality_mconstraint(
            Optimization::inequalitySimplexConstraints, nullptr, tols);


        std::cout << "Optimization dim: " << effective_dim << std::endl << std::endl;

        optimizator.set_min_objective(Optimization::objFunctionWrapper<GIRL<ActionC, StateC>> , this);
        optimizator.set_xtol_rel(1e-8);
        optimizator.set_ftol_rel(1e-8);
        optimizator.set_ftol_abs(1e-8);
        optimizator.set_maxeval(maxFunEvals);

        // define initial point
        if (starting.n_elem == 0)
        {
            starting.ones(effective_dim);
            starting /= arma::sum(starting);
        }
        else
        {
            assert(effective_dim <= starting.n_elem);
        }
        std::vector<double> parameters(effective_dim);
        for (int i = 0; i < effective_dim; ++i)
            parameters[i] = starting[i];

        //optimize dual function
        double minf;
        if (optimizator.optimize(parameters, minf) < 0)
        {
            std::cout << "nlopt failed!" << std::endl;
        }
        else
        {
            std::cout << "found minimum = " << minf << std::endl;

            // reconstruct parameters
            int dim = active_feat.n_elem;
            int n = parameters.size();

            arma::vec x(dpr, arma::fill::zeros);

            if (n == dim - 1)
            {
                // simplex scenario
                double sumx = 0.0;
                for (int i = 0; i < n; ++i)
                {
                    x(active_feat(i)) = parameters[i];
                    sumx += parameters[i];
                }
                x(active_feat(n)) = 1 - sumx;
            }
            else
            {
                // full features
                for (int i = 0; i < dim; ++i)
                {
                    x(active_feat(i)) = parameters[i];
                }
            }

            rewardf.setParameters(x);
        }
    }

    //======================================================================
    // OBJECTIVE FUNCTION
    //----------------------------------------------------------------------

    double objFunction(const arma::vec& x, arma::vec& df)
    {

        ++nbFunEvals;

        // reconstruct parameters
        int dpr = rewardf.getParametersSize();
        int n = x.n_elem;
        arma::vec parV(dpr, arma::fill::zeros);
        int dim = active_feat.n_elem;

        // simplex scenario
        parV(active_feat(arma::span(0, dim - 2))) = x;
        parV(active_feat(dim - 1)) = 1.0 - sum(x);

        // dispatch the right call
        arma::vec gradient = gradientCalculator->computeGradient(parV);
        arma::mat dGradient = gradientCalculator->getGradientDiff();

        //compute objective function and derivative
        double f = arma::as_scalar(gradient.t() * gradient);
        df = 2.0 * dGradient.t() * gradient;

        //compute the derivative wrt active features and symplex
        df = dtheta_simplex * df;

        std::cout << "g2: " << f << std::endl;
        std::cout << "dwdj: " << dGradient;
        std::cout << "df: " << df.t();
        std::cout << "x:  " << x.t();
        std::cout << "x_full:  " << parV.t();
        std::cout << "-----------------------------------------" << std::endl;

        return f;
    }

    //======================================================================
    // PRE-PROCESSING
    //----------------------------------------------------------------------

    void preprocessing()
    {
        // get reward parameters dimension
        int dpr = rewardf.getParametersSize();

        //initially all features are active
        active_feat.set_size(dpr);
        std::iota(std::begin(active_feat), std::end(active_feat), 0);

        // performs preprocessing in order to remove the features
        // that are constant and the one that are almost never
        // under the given samples
        arma::uvec const_ft;
        arma::vec mu = preproc_linear_reward(const_ft);

        // normalize features
        mu = arma::normalise(mu);

        //find non-zero features
        arma::uvec q = arma::find(abs(mu) > 1e-6);

        //sort indexes
        q = arma::sort(q);
        const_ft = arma::sort(const_ft);

        //compute set difference in order to obtain active features
        auto it = std::set_difference(q.begin(), q.end(), const_ft.begin(),
                                      const_ft.end(), active_feat.begin());
        active_feat.resize(it - active_feat.begin());

        std::cout << "=== PRE-PROCESSING ===" << std::endl;
        std::cout << "Feature expectation\n mu: " << mu.t();
        std::cout << "Constant features\n cf: " << const_ft.t();
        std::cout << "Based on mu test, the following features are preserved\n q: " << q.t();
        std::cout << "Finally the active features are\n q - cf: " << active_feat.t();
        std::cout << "=====================================" << std::endl;


        //compute simplex derivative
        computeSimplexDerivative(dpr);

        std::cout << std::endl << "Initial dim: " << dpr << std::endl;
        if (active_feat.n_elem < dpr)
        {
            std::cout << std::endl << "Reduced dim: " << active_feat.n_elem << std::endl;
            std::cout << std::endl << " indicies: " << active_feat.t();
        }
        else
            std::cout << "NO feature reduction!" << std::endl;
    }

    /**
     * @brief Compute the feature expectation and identifies the constant features
     * The function computes the features expectation over trajecteries that can be
     * used to remove the features that are never or rarelly visited under the given
     * samples.
     * Moreover, it identifies the features that are constant. We consider a feature
     * constant when its range (max-min) over an episode is less then a threshold.
     * Clearly, this condition must hold for every episode.
     * @param const_features vector storing the indexis of the constant features
     * @param tol threshold used to test the range
     * @return the feature expectation
     */
    arma::vec preproc_linear_reward(arma::uvec& const_features, double tol =
                                        1e-4)
    {
        int nEpisodes = data.size();
        unsigned int dpr = rewardf.getParametersSize();
        unsigned int dp = policy.getParametersSize();
        arma::vec mu(dpr, arma::fill::zeros);

        arma::mat constant_reward(dpr, nEpisodes, arma::fill::zeros);

        for (int ep = 0; ep < nEpisodes; ++ep)
        {

            //Compute episode J
            int nbSteps = data[ep].size();
            double df = 1.0;

            // store immediate reward over trajectory
            arma::mat reward_vec(dpr, nbSteps, arma::fill::zeros);

            for (int t = 0; t < nbSteps; ++t)
            {
                auto tr = data[ep][t];

                ParametricRegressor& tmp = rewardf;
                reward_vec.col(t) = tmp.diff<StateC, ActionC, StateC>(tr.x, tr.u, tr.xn);

                mu += df * reward_vec.col(t);

                df *= gamma;
            }

            // check reward range over trajectories
            arma::vec R = range(reward_vec, 1);
            for (int p = 0; p < R.n_elem; ++p)
            {
                // check range along each feature
                if (R(p) <= tol)
                {
                    constant_reward(p, ep) = 1;
                }
            }

        }

        const_features = arma::find(arma::sum(constant_reward, 1) == nEpisodes);

        mu /= nEpisodes;

        return mu;

    }

    //======================================================================
    // GETTERS and SETTERS
    //----------------------------------------------------------------------

    unsigned int getFunEvals()
    {
        return nbFunEvals;
    }


private:
    void computeSimplexDerivative(unsigned int dpr)
    {
        unsigned int dim = active_feat.n_elem;
        unsigned int n = dim - 1;

        dtheta_simplex = arma::mat(n, dpr, arma::fill::zeros);

        int i;
        for (i = 0; i < n; i++)
        {
            unsigned int index = active_feat(i);
            dtheta_simplex(i, index) = 1.0;
        }

        dtheta_simplex.col(active_feat(i)) -= arma::ones(n);
    }

protected:
    Dataset<ActionC, StateC>& data;
    DifferentiablePolicy<ActionC, StateC>& policy;
    LinearApproximator& rewardf;
    double gamma;
    IrlGrad aType;

    GradientCalculator<ActionC, StateC>* gradientCalculator;

    unsigned int nbFunEvals;
    arma::uvec active_feat;
    arma::mat dtheta_simplex;

};

} //end namespace

#endif /* GIRL_H_ */
