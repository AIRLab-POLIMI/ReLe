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

#ifndef SRC_FEATURE_SELECTION_FEATURESELECTIONALGORITHM_H_
#define SRC_FEATURE_SELECTION_FEATURESELECTIONALGORITHM_H_


#include <armadillo>

namespace ReLe
{

/*!
 * This class is the basic interface for features selection algorithm.
 * Features selection is the task of computing, from a set of features \f$\phi\f$
 * a new set of features \f$\bar{\phi}\f$, such that:
 * \f$\exists T: \bar{\phi}=T(\phi)\f$
 */
class FeatureSelectionAlgorithm
{
public:
    /*!
     * This method computes the new features from the initial ones
     * \param features the initial features
     */
    virtual void createFeatures(const arma::mat& features) = 0;

    /*!
     * Getter.
     * \return the new features, computed by the algorithm.
     */
    virtual arma::mat getNewFeatures() = 0;

    virtual ~FeatureSelectionAlgorithm()
    {

    }
};

/*!
 * This class is the basic interface for linear features selection algorithms,
 * i.e. a set of algorithms where the features selection functon is a linear function:
 * \f$\bar{\phi}=T\phi\f$
 */
class LinearFeatureSelectionAlgorithm : public FeatureSelectionAlgorithm
{
public:
    /*!
     * Getter.
     * \return the linear features transformation.
     */
    virtual arma::mat getTransformation() = 0;


    virtual ~LinearFeatureSelectionAlgorithm()
    {

    }
};

}



#endif /* SRC_FEATURE_SELECTION_FEATURESELECTIONALGORITHM_H_ */
