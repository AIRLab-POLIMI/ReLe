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

#ifndef MLE_H_
#define MLE_H_

#include <armadillo>
#include "Policy.h"
#include "Transition.h"

#include <nlopt.hpp>
#include <cassert>

namespace ReLe
{

class MLE
{
public:
    MLE(DifferentiablePolicy<DenseAction,DenseState>& policy, Dataset<DenseAction,DenseState>& ds)
        : policy(policy), data(ds)
    {
    }

    arma::vec solve(arma::vec starting = arma::vec(),
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
        optimizator = nlopt::opt(nlopt::algorithm::LD_MMA, dp);
        optimizator.set_max_objective(MLE::wrapper, this);
        optimizator.set_xtol_rel(1e-10);
        optimizator.set_ftol_rel(1e-10);
        optimizator.set_ftol_abs(1e-10);
        optimizator.set_maxeval(maxFunEvals);

        //optimize the function
        std::vector<double> parameters(dp, 0.0);
        for (int i = 0; i < dp; ++i)
            parameters[i] = starting[i];
        double minf;

        // reset function evaluation counter
        nbFunEvals = 0;
        if (optimizator.optimize(parameters, minf) < 0)
        {
            printf("nlopt failed!\n");
            abort();
        }
        else
        {
            printf("found minimum = %0.10g\n", minf);

            arma::vec finalP(dp);
            for(int i = 0; i < dp; ++i)
            {
                finalP(i) = parameters[i];
            }

            return finalP;
        }
    }

    double objFunction(unsigned int n, const double* x, double* grad)
    {
        ++nbFunEvals;

        int dp = policy.getParametersSize();
        assert(dp == n);
        arma::vec params(x, dp);
        policy.setParameters(params);

        int nbEpisodes = data.size();
        double logLikelihood = 0.0;
        int counter = 0;
        arma::vec gradient(dp, arma::fill::zeros);
        for (int ep = 0; ep < nbEpisodes; ++ep)
        {
            int nbSteps = data[ep].size();
            for (int t = 0; t < nbSteps; ++t)
            {
                Transition<DenseAction, DenseState>& tr = data[ep][t];

                // compute probability
                double prob = policy(tr.x,tr.u);
                prob = std::max(1e-8,prob);
                logLikelihood += log(prob);

                // compute gradient
                if (grad != nullptr)
                {
                    gradient += policy.difflog(tr.x, tr.u);
                }

                // increment counter of number of samples
                ++counter;
            }
        }

        // compute average value
        logLikelihood /= counter;
        if (grad != nullptr)
        {
            for (int i = 0; i < dp; ++i)
            {
                grad[i] = gradient(i) / counter;
            }
            std::cout << gradient << std::endl;
        }

        return logLikelihood;
    }


    static double wrapper(unsigned int n, const double* x, double* grad,
                          void* o)
    {
        return reinterpret_cast<MLE*>(o)->objFunction(n, x, grad);
    }

    unsigned int getFunEvals()
    {
        return nbFunEvals;
    }

private:
    DifferentiablePolicy<DenseAction,DenseState>& policy;
    Dataset<DenseAction,DenseState>& data;
    unsigned int nbFunEvals;
};

} //end namespace
#endif //MLE_H_
