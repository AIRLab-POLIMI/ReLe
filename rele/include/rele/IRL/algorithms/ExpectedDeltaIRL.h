/*
 * rele,
 *
 *
 * Copyright (C) 2016 Davide Tateo
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

#ifndef INCLUDE_RELE_IRL_ALGORITHMS_EXPECTEDDELTAIRL_H_
#define INCLUDE_RELE_IRL_ALGORITHMS_EXPECTEDDELTAIRL_H_

#include "rele/IRL/IRLAlgorithm.h"
#include "rele/policy/Policy.h"
#include "rele/core/Transition.h"
#include "rele/IRL/utils/GradientUtils.h"
#include "rele/IRL/utils/hessian/HessianCalculator.h"

#include "rele/IRL/utils/IrlGradType.h"

#include <nlopt.hpp>
#include <cassert>

namespace ReLe
{

template<class ActionC, class StateC>
class ExpectedDeltaIRL: public IRLAlgorithm<ActionC, StateC>
{
public:
    ExpectedDeltaIRL(Dataset<ActionC, StateC>& data,
                     DifferentiablePolicy<ActionC, StateC>& policy,
                     LinearApproximator& rewardf, double gamma, IrlGrad type) :
        data(data), policy(policy), rewardf(rewardf)
    {
        nbFunEvals = 0;

        gradientCalculator = GradientCalculatorFactory<ActionC, StateC>::build(
                                 type, rewardf.getFeatures(), data, policy, gamma);
        hessianCalculator = new HessianCalculator<ActionC, StateC>(
            rewardf.getFeatures(), data, policy, gamma);
    }

    virtual ~ExpectedDeltaIRL()
    {

    }

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
        //preprocessing();

        //compute effective parameters dimension
        int effective_dim = rewardf.getParametersSize();

        // handle the case of only one active reward feature
        // FIXME implement

        //setup optimization algorithm
        nlopt::opt optimizator;
        optimizator = nlopt::opt(nlopt::algorithm::LN_COBYLA, effective_dim);

        std::cout << "Optimization dim: " << effective_dim << std::endl
                  << std::endl;

        optimizator.set_min_objective(ExpectedDeltaIRL::wrapper, this);
        optimizator.set_xtol_rel(1e-8);
        optimizator.set_ftol_rel(1e-8);
        optimizator.set_ftol_abs(1e-8);
        optimizator.set_maxeval(maxFunEvals);

        std::vector<double> tols(effective_dim + 1, 1e-5);
        optimizator.add_inequality_mconstraint(
            ExpectedDeltaIRL::InequalitySimplexConstraints, nullptr,
            tols);

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
            int dim = parameters.size(); //FIXME
            int n = parameters.size();

            arma::vec x(dpr, arma::fill::zeros);

            if (n == dim - 1)
            {
                // simplex scenario
                /*double sumx = 0.0;
                for (int i = 0; i < n; ++i)
                {
                	x(active_feat(i)) = parameters[i];
                	sumx += parameters[i];
                }
                x(active_feat(n)) = 1 - sumx;*/
                //TODO fixme
            }
            else
            {
                // full features
                for (int i = 0; i < dim; ++i)
                {
                    //x(active_feat(i)) = parameters[i];
                    x(i) = parameters[i];
                }
            }

            rewardf.setParameters(x);
        }
    }

    //======================================================================
    // OPTIMIZATION: WRAPPER AND OBJECTIVE FUNCTION
    //----------------------------------------------------------------------
    static double wrapper(unsigned int n, const double* x, double* grad,
                          void* o)
    {
        arma::vec df;
        arma::vec parV(const_cast<double*>(x), n, true);
        double value = static_cast<ExpectedDeltaIRL*>(o)->objFunction(parV, df);

        //Save gradient
        if (grad)
        {
            for (int i = 0; i < df.n_elem; ++i)
            {
                grad[i] = df[i];
            }
        }

        //Print gradient and value
        //printOptimizationInfo(value, n, x, grad);

        return value;
    }

    double objFunction(const arma::vec& x, arma::vec& df)
    {

        ++nbFunEvals;

        arma::vec g = gradientCalculator->computeGradient(x);
        arma::mat H = hessianCalculator->computeHessian(x);
        arma::mat Sigma(H.n_rows, H.n_cols, arma::fill::eye);

        double f_linear = -arma::as_scalar(g.t() * arma::inv(H) * g);
        double f_quadratic = 0.5 * arma::as_scalar(g.t() * arma::inv(H) * g);
        double f_trace = arma::trace(H * Sigma)/1e15;
        double f = f_linear + f_quadratic + f_trace;

        std::cout << "f: " << f << std::endl;
        std::cout << "f_linear: " << f_linear << std::endl;
        std::cout << "f_quadratic: " << f_quadratic << std::endl;
        std::cout << "f_trace: " << f_trace << std::endl;
        std::cout << "-----------------------------------------" << std::endl;

        return f;

    }

    static void InequalitySimplexConstraints(unsigned m, double* result,
            unsigned n, const double* x, double* grad, void* f_data)
    {
        result[n] = -1.0;
        for (unsigned int i = 0; i < n; ++i)
        {
            // -x_i <= 0
            result[i] = -x[i];
            // sum x_i - 1 <= 0
            result[n] += x[i];
        }
        //compute the gradient: d c_i / d x_j = g(i*n+j)
        if (grad != nullptr)
        {
            for (unsigned int j = 0; j < n; ++j)
            {
                for (unsigned int i = 0; i < n; ++i)
                {
                    if (i == j)
                    {
                        grad[i * n + j] = -1.0;
                    }
                    else
                    {
                        grad[i * n + j] = 0.0;
                    }

                }
                grad[n * n + j] = 1.0;
            }
        }
    }

private:
    Dataset<ActionC, StateC>& data;
    DifferentiablePolicy<ActionC, StateC>& policy;
    LinearApproximator& rewardf;

    GradientCalculator<ActionC, StateC>* gradientCalculator;
    HessianCalculator<ActionC, StateC>* hessianCalculator;


    unsigned int nbFunEvals;
};

}

#endif /* INCLUDE_RELE_IRL_ALGORITHMS_EXPECTEDDELTAIRL_H_ */
