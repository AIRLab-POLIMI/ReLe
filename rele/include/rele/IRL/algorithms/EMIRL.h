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

#include "rele/IRL/IRLAlgorithm.h"
#include "rele/policy/Policy.h"
#include "rele/core/Transition.h"
#include "rele/utils/ArmadilloExtensions.h"
#include "rele/IRL/feature_selection/PrincipalFeatureAnalysis.h"

#include <nlopt.hpp>

namespace ReLe
{

template<class ActionC, class StateC>
class EMIRL: public IRLAlgorithm<ActionC, StateC>
{
public:
    EMIRL(Dataset<ActionC, StateC>& data, const arma::mat& theta, const arma::vec& wBar, const arma::mat& sigma,
          LinearApproximator& rewardFunction, double gamma)
        : data(data), rewardFunction(rewardFunction), theta(theta), wBar(wBar), sigmaInv(arma::inv(sigma))
    {
        Features& phi = rewardFunction.getFeatures();
        phiBar = data.computeEpisodeFeatureExpectation(phi, gamma);

        preprocess();

        omega = arma::vec(phi.rows(), arma::fill::zeros);
    }

    virtual void run() override
    {
        unsigned int effective_dim = phiBar.n_rows - 1;

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
            EMIRL::InequalitySimplexConstraints, nullptr, tols);

        optimizator.set_min_objective(EMIRL::wrapper, this);
        optimizator.set_xtol_rel(1e-8);
        optimizator.set_ftol_rel(1e-8);
        optimizator.set_ftol_abs(1e-8);
        optimizator.set_maxeval(600);

        //optimize
        std::vector<double> parameters(effective_dim);
        for (int i = 0; i < effective_dim; ++i)
            parameters[i] = 1.0/phiBar.n_rows;
        double minf = 0;
        if (effective_dim != 0 && optimizator.optimize(parameters, minf) < 0)
        {
            std::cout << "nlopt failed!" << std::endl;
        }
        else
        {
            std::cout << "found minimum = " << minf << std::endl;

            double sumx = 0.0;
            for (int i = 0; i < effective_dim; ++i)
            {
                omega(active_feat(i)) = parameters[i];
                sumx += parameters[i];
            }

            omega(active_feat(effective_dim)) = 1.0 - sumx;

            rewardFunction.setParameters(omega);
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
        printOptimizationInfo(value, n, x, grad);

        return value;
    }

    double objFunction(const arma::vec& x, arma::vec& df)
    {
        //Compute expectation-maximization update
        arma::vec xlast = {1.0 - arma::sum(x)};
        arma::vec omega = arma::join_vert(x, xlast);
        arma::vec Jep = phiBar.t()*omega;
        double maxJep = arma::max(Jep);
        Jep = Jep - maxJep; //Numerical trick
        arma::vec a = arma::exp(Jep);
        a /= arma::sum(a);

        arma::vec what = theta*a;

        //Compute Kullback Leiber divergence
        arma::vec delta = what - wBar;
        double KL = arma::as_scalar(delta.t()*sigmaInv*delta);

        //Compute derivative
        arma::mat dwhat = theta*(arma::diagmat(a) - a*a.t())*phiBar.t();
        arma::vec dKL = 2*dwhat.t()*sigmaInv*delta;

        arma::mat dSimplex = arma::join_horiz(arma::eye(x.n_elem, x.n_elem), -arma::ones(x.n_elem));

        df = dSimplex * dKL;

        std::cout << "-------------------------------------" << std::endl;
        std::cout << "Jep Max" << std::endl << maxJep << std::endl;
        std::cout << "Jep Min" << std::endl << std::endl << arma::min(Jep) << std::endl;
        std::cout << "a Min" << std::endl << a.min() << std::endl;
        std::cout << "a Max" << std::endl << a.max() << std::endl;
        std::cout << "a mean" << std::endl << arma::mean(a) << std::endl;
        std::cout << "dwhat Min" << std::endl << dwhat.min() << std::endl;
        std::cout << "dwhat Max" << std::endl << dwhat.max() << std::endl;
        std::cout << "rank(dwhat) " << std::endl << arma::rank(dwhat) << std::endl;
        std::cout << "delta" << std::endl << delta.t() << std::endl;
        std::cout << "what" << std::endl << what.t() << std::endl;
        std::cout << "dKL" << std::endl << dKL.t() << std::endl;
        std::cout << "df" << std::endl << df.t() << std::endl;

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
    // PREPROCESSING
    //----------------------------------------------------------------------
    void preprocess()
    {
        // performs preprocessing in order to remove the features
        // that are constant and the one that are almost never
        // under the given samples
        active_feat.resize(phiBar.n_cols);

        //check feature range over trajectories
        double tol = 1e-4;
        arma::vec R = range(phiBar, 1);
        arma::uvec const_ft = arma::find(R <= tol);

        // normalize features
        arma::vec mu = arma::sum(phiBar, 1) / phiBar.n_cols;
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

        //Compute reduced features set
        phiBar = phiBar.rows(active_feat);

        std::cout << "=== LINEAR REWARD: PRE-PROCESSING ===" << std::endl;
        std::cout << "Feature expectation\n mu: " << mu.t();
        std::cout << "Constant features\n cf: " << const_ft.t();
        std::cout << "Based on mu test, the following features are preserved\n q: " << q.t();
        std::cout << "Finally the active features are\n q - cf: " << active_feat.t();
        std::cout << "=====================================" << std::endl;


        if(arma::rank(theta*phiBar.t()) == 0)
            std::cout << "=========== WARNING!!! ZERO RANK PRODUCT ============" << std::endl;
    }


    //======================================================================
    // DESTRUCTOR
    //----------------------------------------------------------------------
    virtual ~EMIRL()
    {

    }

protected:
    static void printOptimizationInfo(double value, unsigned int n, const double* x,
                                      double* grad)
    {
        std::cout << "v= " << value << " ";
        std::cout << "x= ";
        for (int i = 0; i < n; i++)
        {
            std::cout << x[i] << " ";
        }
        std::cout << std::endl;
        if (grad)
        {
            std::cout << "g= ";
            for (int i = 0; i < n; i++)
            {
                std::cout << grad[i] << " ";
            }
            std::cout << std::endl;
        }
    }

private:
    Dataset<ActionC, StateC>& data;
    LinearApproximator& rewardFunction;
    arma::mat theta;

    arma::mat phiBar;
    arma::vec wBar;
    arma::mat sigmaInv;

    arma::vec omega;

    arma::uvec active_feat;
};

}

#endif /* INCLUDE_RELE_IRL_ALGORITHMS_EMIRL_H_ */
