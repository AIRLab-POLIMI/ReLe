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

#ifndef INCLUDE_RELE_IRL_ALGORITHMS_EMIRL_H_
#define INCLUDE_RELE_IRL_ALGORITHMS_EMIRL_H_

#include "IRLAlgorithm.h"
#include "Policy.h"
#include "Transition.h"

#include <nlopt.hpp>

namespace ReLe
{

template<class ActionC, class StateC>
class EMIRL: public IRLAlgorithm<ActionC, StateC>
{
public:
    EMIRL(Dataset<ActionC, StateC>& data, const arma::mat& theta, const arma::vec& wBar, const arma::mat& sigma,
          Features& phi, double gamma) //TODO correct Features with template params
        : data(data), theta(theta), phiBar(phi.rows(), data.size()), wBar(wBar), sigmaInv(arma::inv(sigma))
    {
        for(unsigned int i = 0; i < data.size(); i++)
        {
            phiBar.col(i) = data[i].computefeatureExpectation(phi, gamma);
        }

        omega = arma::vec(phi.rows(), arma::fill::ones);
        omega /= phi.rows();
    }

    virtual void run()
    {
    	unsigned int effective_dim = omega.n_elem - 1;

        nlopt::opt optimizator;
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
            EMIRL::InequalitySimplexConstraints, NULL, tols);

        optimizator.set_min_objective(EMIRL::wrapper, this);
        optimizator.set_xtol_rel(1e-8);
        optimizator.set_ftol_rel(1e-8);
        optimizator.set_ftol_abs(1e-8);
        optimizator.set_maxeval(600);

        //optimize
        std::vector<double> parameters(effective_dim);
        for (int i = 0; i < effective_dim; ++i)
            parameters[i] = omega[i];
        double minf;
        if (optimizator.optimize(parameters, minf) < 0)
        {
            std::cout << "nlopt failed!" << std::endl;
        }
        else
        {
            std::cout << "found minimum = " << minf << std::endl;

            double sumx = 0.0;
            for (int i = 0; i < effective_dim; ++i)
            {
                omega(i) = parameters[i];
                sumx += parameters[i];
            }
            omega(effective_dim) = 1.0 - sumx;
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
        double value = static_cast<EMIRL*>(o)->objFunction(parV, df);

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
        //Compute expectation-maximization update
    	arma::vec xlast = {1.0 - arma::sum(x)};
        arma::vec omega = arma::join_vert(x, xlast);
        arma::vec Jep = phiBar.t()*omega;
        arma::vec a = arma::exp(Jep);
        a /= arma::sum(a);

        arma::vec what = theta*a;

        //Compute Kullback Leiber divergence
        arma::vec delta = what - wBar;
        double KL = arma::as_scalar(delta.t()*sigmaInv*delta);

        //Compute derivative
        arma::mat dwhat = theta*(arma::diagmat(a) - a*a.t())*phiBar.t();
        arma::vec dKL = 2*dwhat*sigmaInv*delta;

        arma::mat dSimplex = arma::join_horiz(arma::eye(x.n_elem, x.n_elem), -arma::ones(x.n_elem));

        df = dSimplex * dKL;

        return KL;

    }

    //======================================================================
    // CONSTRAINTS
    //----------------------------------------------------------------------
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

    //======================================================================
    // GETTERS and SETTERS
    //----------------------------------------------------------------------
    virtual arma::vec getWeights()
    {
        return omega;
    }

    virtual Policy<ActionC, StateC>* getPolicy()
    {
        return nullptr; //TODO implement
    }


    //======================================================================
    // DESTRUCTOR
    //----------------------------------------------------------------------

    virtual ~EMIRL()
    {

    }

private:
    Dataset<ActionC, StateC>& data;
    arma::mat theta;

    arma::mat phiBar;
    arma::vec wBar;
    arma::mat sigmaInv;

    arma::vec omega;
};

}

#endif /* INCLUDE_RELE_IRL_ALGORITHMS_EMIRL_H_ */
