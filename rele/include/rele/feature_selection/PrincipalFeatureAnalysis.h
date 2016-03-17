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

#ifndef INCLUDE_RELE_FEATURE_SELECTION_PRINCIPALFEATUREANALYSIS_H_
#define INCLUDE_RELE_FEATURE_SELECTION_PRINCIPALFEATUREANALYSIS_H_


#include "rele/feature_selection/FeatureSelectionAlgorithm.h"

namespace ReLe
{

/*!
 * This class implements the Principal Feature Analysis (PFA).
 * This method search the most promising features index to reduce dimensionality,
 * thus maintaining the initial features, instead of creating new ones.
 */
class PrincipalFeatureAnalysis : public LinearFeatureSelectionAlgorithm
{
public:
    /*!
     * Constructor.
     * \param varMin the minimum variability to retain from data
     * \param useCorrelation if to use correlation or covariance matrix as selection criterion
     */
    PrincipalFeatureAnalysis(double varMin, bool useCorrelation = true);
    virtual void createFeatures(const arma::mat& features) override;
    virtual arma::mat getTransformation() override;
    virtual arma::mat getNewFeatures() override;

    /*!
     * Getter.
     * \return the indexes of the most promising features
     */
    inline arma::uvec getIndexes()
    {
        return indexes;
    }

private:
    unsigned int computeDimensions(arma::vec& s, double varMin);
    void cluster(arma::mat& data, arma::mat& means, arma::uvec& clustersIndexes, unsigned int k);
    unsigned int findNearest(const arma::mat& elements, const arma::vec mean);

private:
    unsigned int initialSize;
    arma::uvec indexes;
    arma::mat newFeatures;

    double varMin;
    bool useCorrelation;

};

}


#endif /* INCLUDE_RELE_FEATURE_SELECTION_PRINCIPALFEATUREANALYSIS_H_ */
