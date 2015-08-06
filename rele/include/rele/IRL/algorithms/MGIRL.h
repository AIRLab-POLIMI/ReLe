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

#ifndef INCLUDE_RELE_IRL_ALGORITHMS_MGIRL_H_
#define INCLUDE_RELE_IRL_ALGORITHMS_MGIRL_H_

#include "algorithms/GIRL.h"

namespace ReLe
{

template<class ActionC, class StateC>
class MGIRL: public GIRL<ActionC, StateC>
{
public:
    MGIRL(Dataset<ActionC, StateC>& dataset,
          DifferentiablePolicy<ActionC, StateC>& policy,
          ParametricRegressor& rewardf, double gamma, IRLGradType aType, bool normalizeGradient = false) :
        GIRL<ActionC, StateC>(dataset, policy, rewardf, gamma, aType), normalizeGradient(normalizeGradient)
    {

    }

    virtual void run()
    {
        run(arma::vec(), 0);
    }

    virtual void run(arma::vec starting, unsigned int maxFunEvals)
    {
        int dpr = this->rewardf.getParametersSize();
        assert(dpr > 0);

        if (starting.n_elem == 0)
        {
            starting.zeros(dpr-1);
        }
        else
        {
            assert(dpr - 1 == starting.n_elem);
        }

        if (maxFunEvals == 0)
            maxFunEvals = std::min(30 * dpr, 600);

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
        optimizator = nlopt::opt(nlopt::algorithm::LD_SLSQP, dpr - 1);

        optimizator.set_min_objective(MGIRL::wrapper, this);
        optimizator.set_xtol_rel(1e-8);
        optimizator.set_ftol_rel(1e-8);
        optimizator.set_ftol_abs(1e-8);
        optimizator.set_maxeval(maxFunEvals);

        //optimize function
        std::vector<double> parameters(dpr - 1);
        for (int i = 0; i < dpr - 1; ++i)
            parameters[i] = starting[i];
        double minf;
        if (optimizator.optimize(parameters, minf) < 0)
        {
            std::cout << "nlopt failed!" << std::endl;
        }
        else
        {
            std::cout << "found minimum = " << minf << std::endl;

            arma::vec finalPreferences(dpr - 1);
            for (int i = 0; i < finalPreferences.size(); ++i)
            {
                finalPreferences(i) = parameters[i];
            }

            arma::vec weights = computeParameters(finalPreferences);
            std::cout << std::endl;

            this->rewardf.setParameters(weights);
        }
    }

    static double wrapper(unsigned int n, const double* x, double* grad,
                          void* o)
    {
        arma::vec df;
        const arma::vec preferences(const_cast<double*>(x), n, true);
        arma::mat dTheta;
        arma::vec parV = computeParameters(preferences, dTheta);
        double value = static_cast<GIRL<ActionC, StateC>*>(o)->objFunction(parV,
                       df);

        //compute the derivative
        arma::vec dM = dTheta.t()*df;

        //normalize to avoid exploding gradients
        if(static_cast<MGIRL*>(o)->normalizeGradient)
        	dM/=arma::norm(df);

        //Save gradient
        if (grad)
        {
            for (int i = 0; i < dM.n_elem; ++i)
            {
                grad[i] = dM[i];
            }
        }

        //Print gradient and value
        //printOptimizationInfo(value, n, x, grad);

        return value;
    }

    virtual ~MGIRL()
    {

    }

private:
    static arma::vec computeParameters(const arma::vec& preferences)
    {
        unsigned int n = preferences.n_elem;
        arma::vec expW(n + 1);
        expW(arma::span(0, n - 1)) = arma::exp(preferences);
        expW(n) = 1;
        return expW / sum(expW);
    }

    static arma::vec computeParameters(const arma::vec& preferences,
                                       arma::mat& dTheta)
    {
        //Compute exponential and l1-norm
        unsigned int n = preferences.n_elem;
        arma::vec expW = arma::exp(preferences);
        double D = arma::sum(expW) + 1;

        //compute derivative
        dTheta = arma::join_vert(D * arma::diagmat(expW) - expW * expW.t(),
                                 -expW.t());
        dTheta /= D * D;

        //compute parameters
        arma::vec theta(n + 1);
        theta(arma::span(0, n - 1)) = expW;
        theta(n) = 1;
        theta /= D;

        return theta;
    }

private:
	bool normalizeGradient;

};

}

#endif /* INCLUDE_RELE_IRL_ALGORITHMS_MGIRL_H_ */
