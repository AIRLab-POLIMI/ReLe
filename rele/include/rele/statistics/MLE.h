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
    MLE(ParametricPolicy<DenseAction,DenseState>& policy, Dataset<DenseAction,DenseState>& ds)
        : policy(policy), data(ds)
    {
    }

    arma::vec solve(arma::vec starting)
    {

        int dp = policy.getParametersSize();
        assert(dp == starting.n_elem);
        nlopt::opt optimizator;
        optimizator = nlopt::opt(nlopt::algorithm::LN_COBYLA, dp);
        optimizator.set_min_objective(MLE::wrapper, this);
        optimizator.set_xtol_rel(1e-6);
        optimizator.set_ftol_rel(1e-6);
        optimizator.set_maxeval(200);

        //optimize the function
        std::vector<double> parameters(dp, 0.0);
        for (int i = 0; i < dp; ++i)
            parameters[i] = starting[i];
        double minf;
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
        int dp = policy.getParametersSize();
        assert(dp == n);
        arma::vec params(x, dp);
        policy.setParameters(params);

        int nbEpisodes = data.size();
        double likelihood = 0.0;
        int counter = 0;
        for (int ep = 0; ep < nbEpisodes; ++ep)
        {
            int nbSteps = data[ep].size();
            for (int t = 0; t < nbSteps; ++t)
            {
                Transition<DenseAction, DenseState>& tr = data[ep][t];
                double prob = policy(tr.x,tr.u);
                prob = std::max(1e-10,prob);
                likelihood += log(prob);

                ++counter;
            }
        }
        likelihood /= counter;
        return -likelihood;
    }


    static double wrapper(unsigned int n, const double* x, double* grad,
                          void* o)
    {
        return reinterpret_cast<MLE*>(o)->objFunction(n, x, grad);
    }

private:
    ParametricPolicy<DenseAction,DenseState>& policy;
    Dataset<DenseAction,DenseState>& data;
};

} //end namespace
#endif //MLE_H_
