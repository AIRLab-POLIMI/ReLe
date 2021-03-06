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

#ifndef INCLUDE_RELE_IRL_ALGORITHMS_SCIRL_H_
#define INCLUDE_RELE_IRL_ALGORITHMS_SCIRL_H_

#include "EpisodicLinearIRLAlgorithm.h"

namespace ReLe
{

template<class StateC>
class SCIRL: public IRLAlgorithm<FiniteAction, StateC>
{
public:
    SCIRL(Dataset<FiniteAction, StateC>& data, LinearApproximator& rewardf,
          double gamma, unsigned int nActions, bool heuristic = true) :
        data(data), rewardf(rewardf), gamma(gamma), nActions(nActions), heuristic(heuristic)
    {
        lambda_mu = 1e-5;
        lambda_c = 0;
        epsilon = 1e-3;
        N_final = 20;
    }

    virtual void run() override
    {
        arma::mat phiData, phiDataNext, phiAll;
        computeFeatures(phiData, phiDataNext, phiAll);
        arma::mat mu = LSTD_mu(phiData, phiDataNext, phiAll);
        classification(mu);
    }

    virtual ~SCIRL()
    {

    }

private:
    void computeFeatures(arma::mat& phiData, arma::mat& phiDataNext, arma::mat& phiAll)
    {
        unsigned int nTransitions = data.getTransitionsNumber() - data.size();
        unsigned int nFeatures = rewardf.getParametersSize();

        Features& phi = rewardf.getFeatures();

        phiData.set_size(nTransitions, nFeatures);
        phiDataNext.set_size(nTransitions, nFeatures);
        phiAll.set_size(nTransitions*nActions, nFeatures);

        unsigned int count = 0;
        for(auto& episode : data)
        {
            for(unsigned t = 0; t + 1 < episode.size(); t++)
            {
                auto& tr = episode[t];
                auto& trNext = episode[t+1];
                phiData.row(count) = phi(tr.x, tr.u).t();
                phiDataNext.row(count) = phi(trNext.x, trNext.u).t();

                for (unsigned int u = 0; u < nActions; u++)
                {
                    phiAll.row(count + nTransitions*u) = phi(tr.x, FiniteAction(u)).t();
                }

                count++;
            }
        }
    }

    arma::mat LSTD_mu(const arma::mat& phiData, const arma::mat& phiDataNext, const arma::mat& phiAll)
    {
        unsigned int nFeatures = rewardf.getParametersSize();

        arma::mat b = phiData.t()*phiData;
        arma::mat A = lambda_mu*arma::eye(nFeatures, nFeatures)+b-gamma*phiData.t()*phiDataNext;
        arma::mat w = arma::solve(A, b);

        if(heuristic)
        {
            unsigned int nTransitions = data.getTransitionsNumber() - data.size();

            arma::mat mu_c = phiData*w;
            arma::mat mu(nTransitions*nActions, nFeatures);

            unsigned int count = 0;
            for(auto& episode : data)
            {
                for(unsigned t = 0; t + 1 < episode.size(); t++)
                {
                    auto& tr = episode[t];

                    for(unsigned int u = 0; u < nActions; u++)
                    {
                        if(u == tr.u)
                            mu.row(count+u) = mu_c.row(count);
                        else
                            mu.row(count+u) = gamma*mu_c.row(count);
                    }

                    count ++;
                }
            }

            return mu;

        }
        else
        {
            return phiAll*w;
        }
    }

    void classification(const arma::mat& mu)
    {
        unsigned int nTransitions = data.getTransitionsNumber() - data.size();
        unsigned int nFeatures = rewardf.getParametersSize();

        arma::vec theta(mu.n_cols, arma::fill::zeros);
        arma::vec margin(mu.n_rows, arma::fill::ones);
        arma::mat phi_sample(nTransitions,nFeatures, arma::fill::zeros);
        arma::mat phi_sample_star(nTransitions,nFeatures, arma::fill::zeros);



        unsigned int count = 0;
        for(auto& episode : data)
        {
            for(unsigned t = 0; t + 1 < episode.size(); t++)
            {
                auto& tr = episode[t];

                margin(count+nTransitions*tr.u)=0;
                phi_sample.row(count)=mu.row(count+nTransitions*tr.u);
                count++;
            }
        }

        double stoppingCondition=1+epsilon;

        unsigned int iterations=0;

        // gradient descend
        while (stoppingCondition > epsilon && iterations < N_final)
        {
            // computing theta derivative
            arma::mat Q_classif=mu*theta+margin;
            arma::uvec a_max;
            max_Q(Q_classif, a_max);

            for (unsigned int i=0; i < nTransitions; i++)
                phi_sample_star.row(i) = mu.row(i+nTransitions*a_max(i));

            arma::mat derivative=arma::ones(1,nTransitions)*(phi_sample_star-phi_sample)/nTransitions+lambda_c*theta.t();
            double delta=1.0/(iterations+1);

            arma::vec oldTheta=theta;
            // update theta
            if (arma::norm(derivative)!=0)
                theta=theta-delta*derivative.t()/(arma::norm(derivative));

            // compute termination criterion
            stoppingCondition=arma::norm(oldTheta-theta);
            iterations++;
        }

        //set weights of the reward function
        rewardf.setParameters(theta);
    }

    void max_Q(const arma::mat& Q, arma::uvec& uMax)
    {
        unsigned int N=Q.n_rows/nActions;
        uMax.set_size(N);

        for (unsigned int i=0; i < N; i++)
        {
            double max = -std::numeric_limits<double>::infinity();

            for(unsigned int u = 0; u < nActions; u++)
            {
                double q = Q(i+N*u);
                if(q > max)
                {
                    max = q;
                    uMax(i) = u;
                }
            }
        }

    }


private:
    Dataset<FiniteAction, StateC>& data;
    LinearApproximator& rewardf;
    double gamma;
    unsigned int nActions;
    bool heuristic;

    double lambda_mu;
    double lambda_c;
    double epsilon;
    unsigned int N_final;
};

}



#endif /* INCLUDE_RELE_IRL_ALGORITHMS_SCIRL_H_ */
