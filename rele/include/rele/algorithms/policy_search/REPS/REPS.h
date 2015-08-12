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

#ifndef REPS_H_
#define REPS_H_

#include "DifferentiableNormals.h"
#include "ArmadilloPDFs.h"
#include "REPSOutputData.h"

#include <nlopt.hpp>
#include "policy_search/BlackBoxAlgorithm.h"

namespace ReLe
{

template<class ActionC, class StateC, class DistributionC>
class REPS: public BlackBoxAlgorithm<ActionC, StateC, DistributionC, REPSOutputData>
{

    USE_BBA_MEMBERS(ActionC, StateC, DistributionC, REPSOutputData)

public:
    REPS(DistributionC& dist, ParametricPolicy<ActionC, StateC>& policy,
         unsigned int nbEpisodes, unsigned int nbPolicies, int reward_obj = 0)
        : BlackBoxAlgorithm<ActionC, StateC, DistributionC, REPSOutputData>
        (dist, policy, nbEpisodes, nbPolicies, reward_obj)
    {
        etaOpt = 1;
        //default parameters
        eps = 0.5;

        maxJ = -std::numeric_limits<double>::infinity();
    }

    virtual ~REPS()
    {}

    inline void setEps(double eps)
    {
        this->eps = eps;
    }

protected:
    virtual void init()
    {
    	theta.set_size(policy.getParametersSize(), nbPoliciesToEvalMetap);
        history_J = arma::vec(nbPoliciesToEvalMetap, arma::fill::zeros);
        maxJ = -std::numeric_limits<double>::infinity();

        //Init policy and parameters
        etaOpt = 1;

        //setup optimization algorithm
        optimizator = nlopt::opt(nlopt::algorithm::LD_MMA, 1);
        optimizator.set_min_objective(REPS::wrapper, this);
        optimizator.set_xtol_rel(1e-8);
        optimizator.set_ftol_rel(1e-12);
        optimizator.set_maxeval(300 * 10);

        std::vector<double> lowerBounds(1, std::numeric_limits<double>::epsilon());
        optimizator.set_lower_bounds(lowerBounds);
    }

    virtual void afterPolicyEstimate()
    {
        //average over episodes
        Jpol /= nbEpisodesToEvalPolicy;
        history_J[polCount] = Jpol;
        if (maxJ < Jpol)
            maxJ = Jpol;
        theta.col(polCount) = policy.getParameters();
    }

    virtual void afterMetaParamsEstimate()
    {
        //--- Update current data
        currentItStats->covariance = dist.getCovariance();
        //---

        //optimize function
        updatePolicy();

        //reset maximum value
        maxJ = -std::numeric_limits<double>::infinity();
    }

    double dualFunction(const double& eta, double& grad)
    {

        double sum1 = 0;
        double sum2 = 0;

        double N = history_J.size();

        for (auto& sample : history_J)
        {
            double r = sample - maxJ; //numeric trick
            sum1 += exp(r / eta);
            sum2 += exp(r / eta) * r;
        }

        sum1 /= N;
        sum2 /= N;

        grad = eps + log(sum1) - sum2 / (eta * sum1);
        return eta * eps + eta * log(sum1) + maxJ;
    }

    static double wrapper(unsigned int n, const double* x, double* grad,
                          void* o)
    {
        return reinterpret_cast<REPS*>(o)->dualFunction(*x, *grad);
    }

    void updatePolicy()
    {
        //optimize dual function
        std::vector<double> parameters(1, etaOpt);
        auto&& newParameters = optimizator.optimize(parameters);

        //update parameters
        etaOpt = newParameters.back();

        //--- save eta value
        currentItStats->eta = etaOpt;
        //---

        //Compute weights
        arma::vec d(history_J.size());
        for (unsigned int i = 0; i < history_J.size(); i++)
        {
            double r = history_J[i] - maxJ;
            d[i] = exp(r / etaOpt);
        }


        dist.wmle(d, theta);

    }

protected:
    arma::mat theta;
    double maxJ;

    double etaOpt;
    double eps;

    nlopt::opt optimizator;

};


} //end namespace

#endif //REPS_H_
