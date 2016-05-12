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
        params.each_col() = getInitialValue();

        double eps = 0.1;
        double posteriorP = -std::numeric_limits<double>::infinity();
        double oldPosteriorP;

        do
        {
            //Reset posterior probability
            oldPosteriorP = posteriorP;

            //Compute policy MAP for each element
            posteriorP = updateTheta(data) / data.size();

            //Compute distribution posterior
            posteriorP += computePosterior();

            //Update theta prior
            computeThetaPrior();

            std::cout << "Posterior: " << posteriorP << std::endl << std::endl;

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
    virtual double updateTheta(const Dataset<ActionC, StateC>& data)
    {
        double posteriorP = 0;

        //Compute policy MAP for each element
        for (unsigned int ep = 0; ep < data.size(); ep++)
        {
            Dataset<ActionC, StateC> epDataset;
            epDataset.push_back(data[ep]);
            MAP<ActionC, StateC> mapCalculator(policy, thetaPrior, epDataset);
            arma::vec theta_ep = params.col(ep);
            double thetaP = mapCalculator.compute(theta_ep);
            posteriorP += thetaP;
            params.col(ep) = policy.getParameters();

            //std::cout << thetaP << std::endl;
        }

        //std::cout << "posteriorTheta " << posteriorP << std::endl;

        return posteriorP;
    }

    virtual double computePosterior() = 0;
    virtual void computeThetaPrior() = 0;
    virtual arma::vec getInitialValue() = 0;

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

    virtual arma::vec getInitialValue() override
    {
        return prior.getMean();
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
                                 const Wishart& precisionPrior) :
        BayesianCoordinateAscend<ActionC, StateC>(policy, meanPrior.getMean(), precisionPrior.getMode().i()),
        meanPrior(meanPrior), precisionPrior(precisionPrior),
        meanPosterior(meanPrior), precisionPosterior(precisionPrior)

    {

    }

    virtual ~BayesianCoordinateAscendFull()
    {

    }

    virtual double computePosterior() override
    {
        //Use previous covariance estimate
        arma::mat precision = precisionPosterior.getMode();
        arma::mat Sigma = precision.i();

        //Compute mean distribution posterior
        meanPosterior = GaussianConjugatePrior::compute(Sigma, meanPrior, this->params);
        arma::vec mu = meanPosterior.getMean();

        //compute covariance distribution posterior
        precisionPosterior = GaussianConjugatePrior::compute(mu, precisionPrior, this->params);

        //Update new covariance estimate
        arma::mat Lambda = precisionPosterior.getMode();

        //compute posterior probability
        double logMuP = meanPosterior.logPdf(mu);
        double logCovP = precisionPosterior.logPdf(arma::vectorise(Lambda));

        std::cout << "posteriorMu " << logMuP << std::endl;
        std::cout << "posteriorCov " << logCovP << std::endl;

        return logMuP + logCovP;
    }

    virtual void computeThetaPrior() override
    {
        this->thetaPrior = ParametricNormal(meanPosterior.getMean(), precisionPosterior.getMode().i());
    }

    ParametricNormal getMeanPosterior()
    {
        return meanPosterior;
    }

    Wishart getPrecisionPosterior()
    {
        return precisionPosterior;
    }

    virtual arma::vec getInitialValue() override
    {
        return meanPrior.getMean();
    }

protected:
    const ParametricNormal& meanPrior;
    const Wishart& precisionPrior;
    ParametricNormal meanPosterior;
    Wishart precisionPosterior;

};


class LinearBayesianCoordinateAscendFull : public BayesianCoordinateAscendFull<DenseAction, DenseState>
{
public:
    LinearBayesianCoordinateAscendFull(MVNPolicy& policy,Features& phi, arma::mat& SigmaPolicy,
                                       const ParametricNormal& meanPrior, const Wishart& precisionPrior)
        :  BayesianCoordinateAscendFull<DenseAction, DenseState>(policy, meanPrior, precisionPrior),
           phi(phi), SigmaInv(SigmaPolicy.i())
    {

    }

protected:
    virtual double updateTheta(const Dataset<DenseAction, DenseState>& data) override
    {
        unsigned int dp = phi.rows();
        double posteriorP = 0;

        //Compute policy MAP for each element
        for (unsigned int ep = 0; ep < data.size(); ep++)
        {
            arma::mat A(dp, dp, arma::fill::zeros);
            arma::vec b(dp, arma::fill::zeros);
            for(auto& tr : data[ep])
            {
                arma::mat phiX = phi(tr.x);
                const arma::vec& u = tr.u;

                A += phiX*SigmaInv*phiX.t();
                b += phiX*SigmaInv*u;
            }

            arma::mat SigmaPriorInv = this->thetaPrior.getCovariance().i();
            arma::vec muPrior = this->thetaPrior.getMean();

            A += SigmaPriorInv;
            b += SigmaPriorInv*muPrior;

            arma::vec p = solve(A, b);
            params.col(ep) = p;

            //compute posterior probability
            double prob =  computePolicyPosterior(p, data[ep]);
            //std::cout << "p " << prob << std::endl;
            posteriorP += prob;

        }

        std::cout << "posteriorTheta " << posteriorP << std::endl;

        return posteriorP;
    }

private:
    double computePolicyPosterior(const arma::vec& p,
                                  const Episode<DenseAction, DenseState>& episode)
    {
        policy.setParameters(p);

        double logLikelihood = 0;
        for (auto& tr : episode)
        {
            double prob = policy(tr.x, tr.u);
            logLikelihood += std::log(prob);
        }

        return logLikelihood + this->thetaPrior.logPdf(p);
    }


private:
    arma::mat SigmaInv;
    Features& phi;
};


}


#endif /* INCLUDE_RELE_IRL_ALGORITHMS_BAYESIANCOORDINATEASCEND_H_ */
