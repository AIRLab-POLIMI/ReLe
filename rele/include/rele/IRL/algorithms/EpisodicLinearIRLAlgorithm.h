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

#ifndef INCLUDE_RELE_IRL_ALGORITHMS_EPISODICLINEARIRLALGORITHM_H_
#define INCLUDE_RELE_IRL_ALGORITHMS_EPISODICLINEARIRLALGORITHM_H_

#include "rele/IRL/algorithms/LinearIRLAlgorithm.h"

namespace ReLe
{

template<class ActionC, class StateC>
class EpisodicLinearIRLAlgorithm : public LinearIRLAlgorithm<ActionC, StateC>
{
public:
    EpisodicLinearIRLAlgorithm(Dataset<ActionC, StateC>& data, const arma::mat& theta,
                               LinearApproximator& rewardf, double gamma)
        : LinearIRLAlgorithm<ActionC, StateC>(data, rewardf, gamma), theta(theta)
    {
        Features& phi = rewardf.getFeatures();
        phiBar = data.computeEpisodeFeatureExpectation(phi, gamma);
    }

    virtual ~EpisodicLinearIRLAlgorithm()
    {

    }

protected:
    //======================================================================
    // PREPROCESSING
    //----------------------------------------------------------------------
    virtual void preprocessing() override
    {
        // performs preprocessing in order to remove the features
        // that are constant and the one that are almost never
        // under the given samples
        unsigned int dpr = phiBar.n_cols;

        arma::uvec active_feat(dpr);

        //check feature range over trajectories
        double tol = 1e-4;
        arma::vec R = range(phiBar, 1);
        arma::uvec const_ft = arma::find(R <= tol);

        // normalize features
        arma::vec mu = arma::mean(phiBar, 1);
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

        std::cout << "=== LINEAR REWARD: PRE-PROCESSING ===" << std::endl;
        std::cout << "Feature expectation\n mu: " << mu.t();
        std::cout << "Constant features\n cf: " << const_ft.t();
        std::cout << "Based on mu test, the following features are preserved\n q: " << q.t();
        std::cout << "Finally the active features are" <<std::endl;
        std::cout << "q - cf: " << active_feat.t();
        std::cout << "=====================================" << std::endl;

        //Compute reduced features set
        if (active_feat.n_elem < dpr)
        {
            std::cout << std::endl << "Reduced dim: " << active_feat.n_elem << std::endl;
            this->simplex.setActiveFeatures(active_feat);
        }
        else
            std::cout << "NO feature reduction!" << std::endl;
    }

protected:
    // Data
    arma::mat theta;
    arma::mat phiBar;

};

}

#endif /* INCLUDE_RELE_IRL_ALGORITHMS_EPISODICLINEARIRLALGORITHM_H_ */
