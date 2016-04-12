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

#ifndef INCLUDE_RELE_POLICY_UTILS_MAP_H_
#define INCLUDE_RELE_POLICY_UTILS_MAP_H_

#include <armadillo>
#include "rele/policy/Policy.h"
#include "rele/core/Transition.h"
#include "rele/optimization/Optimization.h"


#include <nlopt.hpp>
#include <cassert>

namespace ReLe
{

/*!
 * This class can be used to compute the Maximum A Posteriori estimate
 * for a generic parametric stochastic policy.
 */
template<class ActionC, class StateC>
class MAP
{
public:
    MAP(DifferentiablePolicy<ActionC,StateC>& policy,
        const DifferentiableDistribution& prior,
        const Dataset<ActionC,StateC>& ds)
        : policy(policy), prior(prior), data(ds)
    {

    }

    virtual double compute(arma::vec starting = arma::vec(),
                           unsigned int maxFunEvals = 0)
    {
        int dp = policy.getParametersSize();
        assert(dp > 0);

        if (starting.n_elem == 0)
        {
            starting.zeros(dp);
        }
        else
        {
            assert(dp == starting.n_elem);
        }

        if (maxFunEvals == 0)
            maxFunEvals = std::min(30*dp, 600);

        nlopt::opt optimizator;
        optimizator = nlopt::opt(nlopt::algorithm::LD_SLSQP, dp);
        optimizator.set_max_objective(Optimization::objFunctionWrapper<MAP, false>, this);
        optimizator.set_ftol_rel(1e-10);
        optimizator.set_ftol_abs(1e-10);
        optimizator.set_maxeval(maxFunEvals);

        //optimize the function
        std::vector<double> parameters(dp, 0.0);
        for (int i = 0; i < dp; ++i)
            parameters[i] = starting[i];
        double minf;

        if (optimizator.optimize(parameters, minf) < 0)
        {
            throw std::runtime_error("nlopt failed!");
        }
        else
        {
            arma::vec finalP(dp);
            for(int i = 0; i < dp; ++i)
            {
                finalP(i) = parameters[i];
            }

            policy.setParameters(finalP);

            return minf;
        }
    }

    double objFunction(const arma::vec& x, arma::vec& dx)
    {
        policy.setParameters(x);

        unsigned int episodeN = data.size();
        unsigned int dp = policy.getParametersSize();
        double logLikelihood = 0.0;

        for (int ep = 0; ep < episodeN; ++ep)
        {
            int nbSteps = data[ep].size();
            for (int t = 0; t < nbSteps; ++t)
            {
                auto& tr = data[ep][t];

                // compute likelihood
                double likelihood = policy(tr.x,tr.u);
                likelihood = std::max(1e-300,likelihood);
                logLikelihood += log(likelihood);

                // compute gradient
                dx += policy.difflog(tr.x, tr.u);
            }
        }

        // compute average value
        logLikelihood /= episodeN;
        dx /= episodeN;

        //compute prior
        double priorP = prior(x);
        double logPrior = std::log(priorP);
        dx += prior.pointDifflog(x);

        return logLikelihood + logPrior;
    }

    virtual ~MAP()
    {

    }

protected:
    DifferentiablePolicy<ActionC,StateC>& policy;
    const DifferentiableDistribution& prior;
    const Dataset<ActionC,StateC>& data;
};

}


#endif /* INCLUDE_RELE_POLICY_UTILS_MAP_H_ */
