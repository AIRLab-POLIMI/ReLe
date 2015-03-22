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

#include "policy_search/PGPE/PGPE.h"
#include "policy_search/NES/NESOutputData.h"
#include "DifferentiableNormals.h"
#include "ArmadilloPDFs.h"

#include <nlopt.hpp>

namespace ReLe
{

template<class ActionC, class StateC, class DistributionC>
class REPS: public BlackBoxAlgorithm<ActionC, StateC, DistributionC, xNESIterationStats>
{

    typedef BlackBoxAlgorithm<ActionC, StateC, DistributionC, xNESIterationStats> Base;
public:
    REPS(DistributionC& dist, ParametricPolicy<ActionC, StateC>& policy,
         unsigned int nbEpisodes, unsigned int nbPolicies,
         bool baseline = true, int reward_obj = 0)
        : BlackBoxAlgorithm<ActionC, StateC, DistributionC, xNESIterationStats>
        (dist, policy, nbEpisodes, nbPolicies, baseline, reward_obj)
    {
        etaOpt = 1;
        //default parameters
        eps = 0.5;
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
        history_theta.assign(Base::nbPoliciesToEvalMetap, arma::vec(Base::policy.getParametersSize()));
        Base::history_J = arma::vec(Base::nbPoliciesToEvalMetap, arma::fill::zeros);
        maxJ = -std::numeric_limits<double>::infinity();

        //Init policy and parameters
        etaOpt = 1;

        //setup optimization algorithm
        optimizator = nlopt::opt(nlopt::algorithm::LD_MMA, 1);
        optimizator.set_min_objective(REPS::wrapper, this);
        optimizator.set_xtol_rel(0.1);
        optimizator.set_ftol_rel(0.1);

        std::vector<double> lowerBounds(1, std::numeric_limits<double>::epsilon());
        optimizator.set_lower_bounds(lowerBounds);
    }

    virtual void afterPolicyEstimate()
    {
        //average over episodes
        Base::Jpol /= Base::nbEpisodesToEvalPolicy;
        Base::history_J[Base::polCount] = Base::Jpol;
        if (maxJ < Base::Jpol)
            maxJ = Base::Jpol;
        history_theta[Base::polCount] = Base::policy.getParameters();
    }

    virtual void afterMetaParamsEstimate()
    {

        //optimize function
        updatePolicy();

        //update meta distribution
        maxJ = -std::numeric_limits<double>::infinity();

    }

    double dualFunction(const double& eta, double& grad)
    {

        double sum1 = 0;
        double sum2 = 0;

        double N = Base::history_J.size();

        for (auto& sample : Base::history_J)
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

        //Compute weights
        arma::vec d(Base::history_J.size());
        for (unsigned int i = 0; i < Base::history_J.size(); i++)
        {
            double r = Base::history_J[i];
            d[i] = exp(r / etaOpt);
        }

        //Compute weights sums
        double dSum = sum(d);
        double d2Sum = sum(square(d));
        double Z = (dSum*dSum - d2Sum) / dSum;

        unsigned int thethaSize = Base::policy.getParametersSize();

        //Compute mean
        arma::vec mean(thethaSize, arma::fill::zeros);
        for(unsigned int i = 0; i < d.size(); i++)
        {
            const arma::vec& theta = history_theta[i];
            mean += d[i]*theta;
        }

        mean /= dSum;

        //Compute covariance
        arma::mat cov(thethaSize, thethaSize, arma::fill::zeros);
        for(unsigned int i = 0; i < d.size(); i++)
        {
            const arma::vec& theta = history_theta[i];
            arma::vec delta = theta - mean;
            cov += d[i]*delta*delta.t();
        }

        cov /= Z;

        //Update high level policy
        Base::dist.setMeanAndCovariance(mean, cov);

    }

protected:
    std::vector<arma::vec> history_theta;
    double maxJ;

    double etaOpt;
    double eps;

    nlopt::opt optimizator;

};


} //end namespace

#endif //REPS_H_
