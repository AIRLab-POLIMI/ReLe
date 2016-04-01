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

#ifndef INCLUDE_RELE_IRL_ALGORITHMS_LINEARIRLALGORITHM_H_
#define INCLUDE_RELE_IRL_ALGORITHMS_LINEARIRLALGORITHM_H_

#include "rele/core/Transition.h"

#include "rele/IRL/IRLAlgorithm.h"
#include "rele/utils/ArmadilloExtensions.h"

#include <nlopt.hpp>
#include "rele/optimization/Optimization.h"
#include "rele/optimization/Simplex.h"

namespace ReLe
{

template<class ActionC, class StateC>
class LinearIRLAlgorithm : public IRLAlgorithm<ActionC, StateC>
{

public:
    LinearIRLAlgorithm(Dataset<ActionC, StateC>& dataset,
                       LinearApproximator& rewardf,
                       double gamma) :
        data(dataset), rewardf(rewardf), gamma(gamma), simplex(rewardf.getParametersSize())
    {
        optAlg = nlopt::algorithm::LD_SLSQP;
        nbFunEvals = 0;
    }

    //======================================================================
    // RUNNERS
    //----------------------------------------------------------------------
    virtual void run() override
    {
        run(arma::vec(), 0);
    }

    void run(arma::vec starting, unsigned int maxFunEvals)
    {
        int dpr = rewardf.getParametersSize();
        assert(dpr > 0);

        if (maxFunEvals == 0)
            maxFunEvals = std::min(30 * dpr, 600);

        nbFunEvals = 0;

        // initialize active features set
        preprocessing();

        //compute effective parameters dimension
        int effective_dim = simplex.getEffectiveDim();
        std::cout << "Optimization dim: " << effective_dim << std::endl << std::endl;

        // handle the case of only one active reward feature
        if (effective_dim == 0)
        {
            rewardf.setParameters(simplex.reconstruct());
            return;
        }

        // setup optimization algorithm
        nlopt::opt optimizator;

        // optimization
        optimizator = nlopt::opt(optAlg, effective_dim);
        optimizator.set_min_objective(
            Optimization::objFunctionWrapper<LinearIRLAlgorithm<ActionC, StateC>, false> , this);
        optimizator.add_inequality_constraint(Optimization::oneSumConstraint, nullptr, 0);

        std::vector<double> lowerBounds(effective_dim, 0.0);
        std::vector<double> upperBounds(effective_dim, 1.0);
        optimizator.set_lower_bounds(lowerBounds);
        optimizator.set_upper_bounds(upperBounds);

        // temination conditions
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

        //optimize function
        double minf;
        if (optimizator.optimize(parameters, minf) < 0)
        {
            std::cout << "nlopt failed!" << std::endl;
        }
        else
        {
            std::cout << "found minimum = " << minf << std::endl;
            arma::vec x = simplex.reconstruct(parameters);
            rewardf.setParameters(x);
        }
    }

    virtual double objFunction(const arma::vec& xSimplex, arma::vec& df) = 0;

    virtual ~LinearIRLAlgorithm()
    {

    }

protected:
    virtual void preprocessing() = 0;

    //======================================================================
    // GETTERS and SETTERS
    //----------------------------------------------------------------------

    unsigned int getFunEvals()
    {
        return nbFunEvals;
    }

protected:
    // Data
    Dataset<ActionC, StateC>& data;
    LinearApproximator& rewardf;
    double gamma;

    // optimization stuff
    unsigned int nbFunEvals;
    Simplex simplex;
    nlopt::algorithm optAlg;


};


}


#endif /* INCLUDE_RELE_IRL_ALGORITHMS_LINEARIRLALGORITHM_H_ */
