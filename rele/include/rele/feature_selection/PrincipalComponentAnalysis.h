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

#ifndef INCLUDE_RELE_IRL_FEATURE_SELECTION_PRINCIPALCOMPONENTANALYSIS_H_
#define INCLUDE_RELE_IRL_FEATURE_SELECTION_PRINCIPALCOMPONENTANALYSIS_H_

#include "rele/feature_selection/FeatureSelectionAlgorithm.h"

namespace ReLe
{

/*!
 * This class implements Principal Component Analysis (PCA).
 * PCA computes a linear combination of a set of features to reduce data dimensionality.
 */
class PrincipalComponentAnalysis : public LinearFeatureSelectionAlgorithm
{
public:
    /*!
     * Constructor.
     * \param k the final number of features to be used
     * \param useCorrelation if to use correlation or covariance matrix as selection criterion
     */
    PrincipalComponentAnalysis(unsigned int k, bool useCorrelation = true);
    virtual void createFeatures(const arma::mat& features) override;
    virtual arma::mat getTransformation() override;
    virtual arma::mat getNewFeatures() override;

private:
    arma::mat newFeatures;
    arma::mat T;

    unsigned int k;
    bool useCorrelation;

};

}

#endif /* INCLUDE_RELE_IRL_FEATURE_SELECTION_PRINCIPALCOMPONENTANALYSIS_H_ */
