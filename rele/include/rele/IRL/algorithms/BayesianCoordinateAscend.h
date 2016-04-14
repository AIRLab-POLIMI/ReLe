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

#ifndef INCLUDE_RELE_IRL_ALGORITHMS_BAYESIANCOORDINATEASCEND_H_
#define INCLUDE_RELE_IRL_ALGORITHMS_BAYESIANCOORDINATEASCEND_H_

#include "rele/core/Transition.h"
#include "rele/statistics/inference/GaussianConjugatePrior.h"
#include "rele/policy/utils/MAP.h"

namespace ReLe
{

template<class ActionC, class StateC>
class BayesianCoordinateAscend
{
public:
    BayesianCoordinateAscend(DifferentiablePolicy<ActionC, StateC>& policy,
                             const arma::vec& mu0, const arma::mat& Sigma0)
        : policy(policy), thetaPrior(mu0, Sigma0)
    {

    }

    virtual ~BayesianCoordinateAscend()
    {

    }

    void compute(const Dataset<ActionC, StateC>& data)
    {
        unsigned int dp = policy.getParametersSize();
        unsigned int n = data.size();

        params.zeros(dp, n);

        double eps = 1e-8;
        double posteriorP = -std::numeric_limits<double>::infinity();
        double oldPosteriorP;

        do
        {
            //Reset posterior probability
            oldPosteriorP = posteriorP;

            //Compute policy MAP for each element
            posteriorP = updateTheta(data);

            //Compute distribution posterior
            posteriorP += computePosterior();

            //Update theta prior
            computeThetaPrior();

        }
        while(posteriorP - oldPosteriorP > eps);
    }

    arma::mat getParameters()
    {
        return params;
    }

    ParametricNormal getDistribution()
    {
        return thetaPrior;
    }


protected:
    double updateTheta(const Dataset<ActionC, StateC>& data)
    {
        double posteriorP = 0;

        //Compute policy MAP for each element
        for (unsigned int ep = 0; ep < data.size(); ep++)
        {
            Dataset<ActionC, StateC> epDataset;
            epDataset.push_back(data[ep]);
            MAP<ActionC, StateC> mapCalculator(policy, thetaPrior, epDataset);
            arma::vec theta_ep = params.col(ep);
            posteriorP += mapCalculator.compute(theta_ep);
            params.col(ep) = policy.getParameters();
        }

        return posteriorP;
    }

    virtual double computePosterior() = 0;
    virtual void computeThetaPrior() = 0;

protected:
    arma::mat params;
    DifferentiablePolicy<ActionC, StateC>& policy;
    ParametricNormal thetaPrior;

};

template<class ActionC, class StateC>
class BayesianCoordinateAscendMean : public BayesianCoordinateAscend<ActionC, StateC>
{
public:
    BayesianCoordinateAscendMean(DifferentiablePolicy<ActionC, StateC>& policy,
                                 const ParametricNormal& prior,
                                 const arma::mat& Sigma) :
        BayesianCoordinateAscend<ActionC, StateC>(policy, prior.getMean(), Sigma),
        Sigma(Sigma), prior(prior), posterior(policy.getParametersSize())
    {

    }

    virtual ~BayesianCoordinateAscendMean()
    {

    }

    virtual double computePosterior() override
    {
        //Compute distribution posterior
        posterior = GaussianConjugatePrior::compute(Sigma, prior, this->params);

        //compute posterior probability
        arma::vec omega = posterior.getMean();
        return posterior.logPdf(omega);
    }

    virtual void computeThetaPrior() override
    {
        this->thetaPrior = ParametricNormal(posterior.getMean(), Sigma);
    }

    ParametricNormal getPosterior()
    {
        return posterior;
    }

private:
    const ParametricNormal& prior;
    ParametricNormal posterior;
    const arma::mat& Sigma;

};

template<class ActionC, class StateC>
class BayesianCoordinateAscendFull : public BayesianCoordinateAscend<ActionC, StateC>
{
public:
    BayesianCoordinateAscendFull(DifferentiablePolicy<ActionC, StateC>& policy,
                                 const ParametricNormal& meanPrior,
                                 const InverseWishart& covPrior) :
        BayesianCoordinateAscend<ActionC, StateC>(policy, meanPrior.getMean(), covPrior.getMode()),
        meanPrior(meanPrior), covPrior(covPrior),
        meanPosterior(policy.getParametersSize()), covPosterior(policy.getParametersSize())

    {

    }

    virtual ~BayesianCoordinateAscendFull()
    {

    }

    virtual double computePosterior() override
    {
        //Use previous covariance estimate
        arma::mat Sigma = covPosterior.getMode();

        //Compute mean distribution posterior
        meanPosterior = GaussianConjugatePrior::compute(Sigma, meanPrior, this->params);
        arma::vec mu = meanPosterior.getMean();

        //compute covariance distribution posterior
        covPosterior = GaussianConjugatePrior::compute(mu, covPrior, this->params);

        //Update new covariance estimate
        Sigma = covPosterior.getMode();

        //compute posterior probability
        return meanPosterior.logPdf(mu) + covPosterior.logPdf(arma::vectorise(Sigma));
    }

    virtual void computeThetaPrior() override
    {
        this->thetaPrior = ParametricNormal(meanPosterior.getMean(), covPosterior.getMode());
    }

    ParametricNormal getMeanPosterior()
    {
        return meanPosterior;
    }

    InverseWishart getCovPosterior()
    {
        return covPosterior;
    }

private:
    const ParametricNormal& meanPrior;
    const InverseWishart& covPrior;
    ParametricNormal meanPosterior;
    InverseWishart covPosterior;

};



}


#endif /* INCLUDE_RELE_IRL_ALGORITHMS_BAYESIANCOORDINATEASCEND_H_ */
