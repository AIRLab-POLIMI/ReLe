/*
 * rele,
 *
 *
 * Copyright (C) 2015 Davide Tateo
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

#ifndef INCLUDE_RELE_IRL_ALGORITHMS_NOGIRL_H_
#define INCLUDE_RELE_IRL_ALGORITHMS_NOGIRL_H_

#include "GIRL.h"

namespace ReLe
{

template <class ActionC, class StateC>
class NoGIRL :public GIRL<ActionC, StateC>
{
public:
    NoGIRL(Dataset<ActionC, StateC>& dataset,
           DifferentiablePolicy<ActionC, StateC>& policy,
           ParametricRegressor& rewardf, double gamma, IRLGradType aType,
		   std::vector<double>& lowerBounds, std::vector<double>& upperBounds) :
        GIRL<ActionC, StateC>(dataset, policy, rewardf, gamma, aType),
        upperBounds(upperBounds), lowerBounds(lowerBounds)
    {

    }

    virtual ~NoGIRL() { }

    virtual void run()
    {
        run(arma::vec(), 0);
    }

    virtual void run(arma::vec starting,
                     unsigned int maxFunEvals)
    {
        int dpr = this->rewardf.getParametersSize();
        assert(dpr > 0);

        if (starting.n_elem == 0)
        {
            starting.ones(dpr);
            starting /= arma::sum(starting);
        }
        else
        {
            assert(dpr == starting.n_elem);
        }

        if (maxFunEvals == 0)
            maxFunEvals = std::min(30*dpr, 600);

        this->nbFunEvals = 0;

        this->maxSteps = 0;
        int nbEpisodes = this->data.size();
        for (int i = 0; i < nbEpisodes; ++i)
        {
            int nbSteps = this->data[i].size();
            if (this->maxSteps < nbSteps)
            	this->maxSteps = nbSteps;
        }


        //setup optimization algorithm
        nlopt::opt optimizator;
        nlopt::opt localOptimizator;
        optimizator = nlopt::opt(nlopt::algorithm::GD_MLSL_LDS, dpr);
        optimizator.set_lower_bounds(lowerBounds);
        optimizator.set_upper_bounds(upperBounds);

        localOptimizator = nlopt::opt(nlopt::algorithm::LD_SLSQP, dpr);

        optimizator.set_min_objective(GIRL<ActionC, StateC>::wrapper, this);
        optimizator.set_xtol_rel(1e-8);
        optimizator.set_ftol_rel(1e-8);
        optimizator.set_ftol_abs(1e-8);
        optimizator.set_maxeval(20);
        optimizator.set_local_optimizer(localOptimizator);

        //optimize dual function
        std::vector<double> parameters(dpr);
        for (int i = 0; i < dpr; ++i)
            parameters[i] = starting[i];
        double minf;
        if (optimizator.optimize(parameters, minf) < 0)
        {
            std::cout << "nlopt failed!" << std::endl;
        }
        else
        {
            //refine result
            localOptimizator.set_xtol_abs(1e-20);
            localOptimizator.set_ftol_abs(1e-20);
            localOptimizator.set_min_objective(GIRL<ActionC, StateC>::wrapper, this);

            localOptimizator.optimize(parameters, minf);

            std::cout << "found minimum = " << minf << std::endl;

            arma::vec finalP(dpr);
            for(int i = 0; i < dpr; ++i)
            {
                finalP(i) = parameters[i];
            }
            std::cout << std::endl;

            this->rewardf.setParameters(finalP);
        }
    }

private:
    std::vector<double>& upperBounds;
    std::vector<double>& lowerBounds;


};


}


#endif /* INCLUDE_RELE_IRL_ALGORITHMS_NOGIRL_H_ */
