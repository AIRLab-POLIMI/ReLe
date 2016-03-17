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

#ifndef INCLUDE_RELE_IRL_ALGORITHMS_STEPBASEDLINEARIRLALGORITHM_H_
#define INCLUDE_RELE_IRL_ALGORITHMS_STEPBASEDLINEARIRLALGORITHM_H_

#include "rele/IRL/algorithms/LinearIRLAlgorithm.h"

namespace ReLe
{

template<class ActionC, class StateC>
class StepBasedLinearIRLAlgorithm : public LinearIRLAlgorithm<ActionC, StateC>
{

public:
    StepBasedLinearIRLAlgorithm(Dataset<ActionC, StateC>& dataset,
                                LinearApproximator& rewardf,
                                double gamma) :
        LinearIRLAlgorithm<ActionC, StateC>(dataset, rewardf, gamma)
    {

    }

    virtual ~StepBasedLinearIRLAlgorithm()
    {

    }

protected:
    virtual void preprocessing() override
    {
        // get reward parameters dimension
        int dpr = this->rewardf.getParametersSize();

        //initially all features are active
        arma::uvec active_feat(dpr);
        std::iota(std::begin(active_feat), std::end(active_feat), 0);

        // performs preprocessing in order to remove the features
        // that are constant and the one that are almost never
        // under the given samples
        arma::uvec const_ft;
        arma::vec mu = preproc_linear_reward(const_ft);

        // normalize features
        mu = arma::normalise(mu);

        //find non-zero features
        arma::uvec q = arma::find(abs(mu) > 1e-6);

        //sort indexes
        q = arma::sort(q);
        const_ft = arma::sort(const_ft);

        //compute set difference in order to obtain active features
        auto it = std::set_difference(q.begin(), q.end(), const_ft.begin(),
                                      const_ft.end(), active_feat.begin());
        active_feat.resize(it - active_feat.begin());

        std::cout << "=== PRE-PROCESSING ===" << std::endl;
        std::cout << "Feature expectation\n mu: " << mu.t();
        std::cout << "Constant features\n cf: " << const_ft.t();
        std::cout << "Based on mu test, the following features are preserved\n q: " << q.t();
        std::cout << "Finally the active features are" << std::endl;
        std::cout << "q - cf: " << active_feat.t();
        std::cout << "=====================================" << std::endl;

        std::cout << std::endl << "Initial dim: " << dpr << std::endl;
        if (active_feat.n_elem < dpr)
        {
            std::cout << std::endl << "Reduced dim: " << active_feat.n_elem << std::endl;
            this->simplex.setActiveFeatures(active_feat);
        }
        else
            std::cout << "NO feature reduction!" << std::endl;
    }


private:
    arma::vec preproc_linear_reward(arma::uvec& const_features, double tol =
                                        1e-4)
    {
        int nEpisodes = this->data.size();
        unsigned int dpr = this->rewardf.getParametersSize();

        arma::vec mu(dpr, arma::fill::zeros);

        arma::mat constant_reward(dpr, nEpisodes, arma::fill::zeros);

        for (int ep = 0; ep < nEpisodes; ++ep)
        {

            //Compute episode J
            int nbSteps = this->data[ep].size();
            double df = 1.0;

            // store immediate reward over trajectory
            arma::mat reward_vec(dpr, nbSteps, arma::fill::zeros);

            for (int t = 0; t < nbSteps; ++t)
            {
                auto tr = this->data[ep][t];

                ParametricRegressor& tmp = this->rewardf;
                reward_vec.col(t) = tmp.diff<StateC, ActionC, StateC>(tr.x, tr.u, tr.xn);

                mu += df * reward_vec.col(t);

                df *= this->gamma;
            }

            // check reward range over trajectories
            arma::vec R = range(reward_vec, 1);
            for (int p = 0; p < R.n_elem; ++p)
            {
                // check range along each feature
                if (R(p) <= tol)
                {
                    constant_reward(p, ep) = 1;
                }
            }

        }

        const_features = arma::find(arma::sum(constant_reward, 1) == nEpisodes);

        mu /= nEpisodes;

        return mu;

    }
};

}


#endif /* INCLUDE_RELE_IRL_ALGORITHMS_STEPBASEDLINEARIRLALGORITHM_H_ */
