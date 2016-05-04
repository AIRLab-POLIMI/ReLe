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

#ifndef INCLUDE_RELE_IRL_ALGORITHMS_TRADEOFFEEGIRL_H_
#define INCLUDE_RELE_IRL_ALGORITHMS_TRADEOFFEEGIRL_H_

#include "rele/IRL/utils/EpisodicGradientCalculatorFactory.h"
#include "EpisodicLinearIRLAlgorithm.h"

namespace ReLe
{

template<class ActionC, class StateC>
class TradeoffEGIRL: public EGIRL<ActionC, StateC>
{
public:
    TradeoffEGIRL(Dataset<ActionC, StateC>& data, const arma::mat& theta, DifferentiableDistribution& dist,
                  LinearApproximator& rewardf, double gamma, IrlEpGrad gtype, IrlEpHess htype, double lambda = 1.0)
        : EGIRL<ActionC, StateC>(data, theta, dist, rewardf, gamma, gtype), htype(htype), lambda(lambda)
    {
        hessianCalculator = EpisodicHessianCalculatorFactory<ActionC, StateC>::build(htype, theta, this->phi, dist, gamma);

        std::cout << "positive definite features" << std::endl;
        arma::cube diff = hessianCalculator->getHessianDiff();
        for(unsigned int i = 0; i < diff.n_slices; i++)
        {
            double maxEig = arma::max(arma::eig_sym(diff.slice(i)));

            if(maxEig < 0)
                std::cout << maxEig << " " << i << std::endl;
        }
    }

    virtual ~TradeoffEGIRL()
    {

    }

    virtual void run() override
    {
        EGIRL<ActionC, StateC>::run();

        arma::vec w = this->rewardf.getParameters();

        arma::mat hessian = this->hessianCalculator->computeHessian(w);
        arma::vec eigenvalues = arma::eig_sym(hessian);

        std::cout << "eigenvalues" << std::endl;
        std::cout << eigenvalues.t() << std::endl;
        std::cout << "slack: " << -arma::max(eigenvalues) << std::endl;

    }

protected:
    virtual void setStartingPoint(arma::vec& starting, unsigned int effective_dim) override
    {
        effective_dim++;

        if (starting.n_elem == 0)
        {
            starting.ones(effective_dim);
            starting(effective_dim-1) = 0;
            starting /= arma::sum(starting);
        }

        // Check if initial solution is already feasible
        if(sdConstraint(effective_dim, starting.mem, nullptr, this) < 0)
        {
            std::cout << "The starting point is already feasible" << std::endl;
            return;
        }
        else
        {
            std::cout << "Searching feasible starting point" << std::endl;
        }

        // Setup optimization
        nlopt::opt optimizator(nlopt::LD_SLSQP, effective_dim);
        optimizator.set_min_objective(sdConstraint, this);
        indexes = arma::cumsum(arma::uvec(effective_dim-1, arma::fill::ones)) -1;
        optimizator.add_inequality_constraint(Optimization::oneSumConstraintIndex, &indexes, 0);

        std::vector<double> lowerBounds(effective_dim, 0.0);
        std::vector<double> upperBounds(effective_dim, 1.0);
        upperBounds[effective_dim -1] = 0;
        optimizator.set_lower_bounds(lowerBounds);
        optimizator.set_upper_bounds(upperBounds);

        // temination conditions
        optimizator.set_xtol_rel(1e-3);
        optimizator.set_ftol_rel(1e-3);
        optimizator.set_ftol_abs(1e-3);
        optimizator.set_maxeval(1000);


        // try to find a feasible solution
        std::vector<double> parameters(effective_dim);
        for (int i = 0; i < effective_dim -1; i++)
            parameters[i] = starting[i];

        parameters[effective_dim - 1] = 0;

        // optimize function
        double minf;

        if (optimizator.optimize(parameters, minf) < 0)
        {
            throw std::runtime_error("Nlopt failed!");
        }

        if(minf > 0)
        {
            std::cout << "WARNING! unable to find feasible starting point" << std::endl;
        }


        starting = arma::conv_to<arma::vec>::from(parameters);

    }

    virtual double objFunction(const arma::vec& x, arma::vec& df) override
    {
        arma::vec xSimplex = x.head(x.n_elem -1);
        arma::vec dfSimplex;
        double value = EGIRL<ActionC, StateC>::objFunction(xSimplex, dfSimplex);

        df.head(df.n_elem -1) = dfSimplex;
        df(df.n_elem -1) = -lambda;

        return value - lambda*x(x.n_elem -1);
    }

    virtual void setupOptimization(unsigned int effective_dim, unsigned int maxFunEvals) override
    {
        this->optAlg = nlopt::LD_SLSQP;
        this->optimizator = nlopt::opt(this->optAlg, effective_dim+1);
        this->optimizator.set_min_objective(
            Optimization::objFunctionWrapper<LinearIRLAlgorithm<ActionC, StateC>, false> , this);

        indexes = arma::cumsum(arma::uvec(effective_dim, arma::fill::ones)) -1;
        this->optimizator.add_inequality_constraint(Optimization::oneSumConstraintIndex, &indexes , 0);

        std::vector<double> lowerBounds(effective_dim+1, 0.0);
        std::vector<double> upperBounds(effective_dim+1, 1.0);
        upperBounds[effective_dim] = std::numeric_limits<double>::infinity();
        this->optimizator.set_lower_bounds(lowerBounds);
        this->optimizator.set_upper_bounds(upperBounds);

        // temination conditions
        this->optimizator.set_xtol_rel(1e-8);
        this->optimizator.set_ftol_rel(1e-8);
        this->optimizator.set_ftol_abs(1e-8);
        this->optimizator.set_maxeval(maxFunEvals);

        this->optimizator.add_inequality_constraint(sdConstraint, this, 1e-10);
    }

    static double sdConstraint(unsigned int n, const double *x,
                               double *grad, void *data)
    {
        auto& self = *static_cast<TradeoffEGIRL<ActionC,StateC>*>(data);

        arma::vec parV(const_cast<double*>(x), n, true);
        arma::vec&& w = self.simplex.reconstruct(parV.head(n-1));

        arma::mat hessian = self.hessianCalculator->computeHessian(w);

        arma::vec lambda;
        arma::mat V;
        arma::eig_sym(lambda, V, hessian);

        double lambdaMax = arma::as_scalar(lambda.tail(1));
        arma::vec vi = V.tail_cols(1);

        if(grad)
        {
            arma::cube diff = self.hessianCalculator->getHessianDiff();
            unsigned int indx_n = self.simplex.getFeatureIndex(n-1);
            arma::mat diff_n = diff.slice(indx_n);
            for(unsigned int i = 0; i < n-1; i++)
            {
                unsigned int indx = self.simplex.getFeatureIndex(i);
                arma::mat diff_i = diff.slice(indx);
                grad[i] = arma::as_scalar(vi.t()*(diff_i-diff_n)*vi);
            }

            grad[n-1] = 1;
        }

        std::cout << "current constraints " << lambdaMax+x[n-1] << std::endl;

        return lambdaMax+x[n-1];
    }

protected:
    IrlEpHess htype;
    double lambda;
    HessianCalculator<ActionC, StateC>* hessianCalculator;
    arma::uvec indexes;
};



}




#endif /* INCLUDE_RELE_IRL_ALGORITHMS_TRADEOFFEEGIRL_H_ */
