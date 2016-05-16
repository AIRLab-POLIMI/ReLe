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
            updateTheta(data);

            //Update theta prior
            computeThetaPrior();

            //Evaluate posterior probability
            posteriorP = computeLogPosterior(data);

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
    virtual void updateTheta(const Dataset<ActionC, StateC>& data)
    {
        //Compute policy MAP for each element
        for (unsigned int ep = 0; ep < data.size(); ep++)
        {
            Dataset<ActionC, StateC> epDataset;
            epDataset.push_back(data[ep]);
            MAP<ActionC, StateC> mapCalculator(policy, thetaPrior, epDataset);
            arma::vec theta_ep = params.col(ep);
            mapCalculator.compute(theta_ep);
            params.col(ep) = policy.getParameters();
        }
    }

    virtual double computeLogPosterior(const Dataset<ActionC, StateC>& data)
    {
        double policyP = computePoliciesPosterior(data);
        double priorP = computePriorProbability();
        double posteriorP = policyP + priorP;

        std::cout << "policyP" << policyP << std::endl;
        std::cout << "priorP" << priorP << std::endl;
        std::cout << "posteriorP" << posteriorP << std::endl;

        return posteriorP;
    }

    virtual void computeThetaPrior() = 0;
    virtual double computePriorProbability() = 0;

    virtual arma::vec getInitialValue() = 0;

private:
    double computePoliciesPosterior(const Dataset<ActionC, StateC>& data)
    {
        double policyP = 0;
        for (unsigned int i = 0; i < params.n_cols; i++)
        {
            auto p = params.col(i);
            auto ep = data[i];
            policyP += computePolicyPosterior(p, ep);
        }
        return policyP;
    }

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
        Sigma(Sigma), prior(prior)
    {

    }

    virtual ~BayesianCoordinateAscendMean()
    {

    }

protected:
    virtual arma::vec getInitialValue() override
    {
        return prior.getMean();
    }

    virtual void computeThetaPrior() override
    {
        auto posterior = GaussianConjugatePrior::compute(Sigma, prior, this->params);

        this->thetaPrior = ParametricNormal(posterior.getMean(), Sigma);
    }

    virtual double computePriorProbability() override
    {
        arma::mat mu = this->thetaPrior.getMean();

        double priorP = prior.logPdf(mu);

        return priorP;
    }

private:
    const ParametricNormal& prior;
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
        meanPrior(meanPrior), precisionPrior(precisionPrior)

    {

    }

    virtual ~BayesianCoordinateAscendFull()
    {

    }

protected:
    virtual arma::vec getInitialValue() override
    {
        return meanPrior.getMean();
    }

    virtual void computeThetaPrior() override
    {
        //Use previous covariance estimate
        arma::mat Sigma = this->thetaPrior.getCovariance();

        //Compute mean distribution posterior
        auto meanPosterior = GaussianConjugatePrior::compute(Sigma, meanPrior, this->params);
        arma::vec mu = meanPosterior.getMean();

        //compute covariance distribution posterior
        auto precisionPosterior = GaussianConjugatePrior::compute(mu, precisionPrior, this->params);

        this->thetaPrior = ParametricNormal(meanPosterior.getMean(), precisionPosterior.getMode().i());
    }

    virtual double computePriorProbability() override
    {
        arma::mat mu = this->thetaPrior.getMean();
        arma::mat Sigma = this->thetaPrior.getCovariance();

        double meanPriorP = meanPrior.logPdf(mu);
        double precisionPriorP = precisionPrior.logPdf(arma::vectorise(Sigma.i()));

        return meanPriorP+precisionPriorP;
    }


protected:
    const ParametricNormal& meanPrior;
    const Wishart& precisionPrior;

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
    virtual void updateTheta(const Dataset<DenseAction, DenseState>& data) override
    {
        unsigned int dp = phi.rows();

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
        }
    }

private:
    arma::mat SigmaInv;
    Features& phi;
};


}


#endif /* INCLUDE_RELE_IRL_ALGORITHMS_BAYESIANCOORDINATEASCEND_H_ */
