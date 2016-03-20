/*
 * rele,
 *
 *
 * Copyright (C) 2015 Davide Tateo
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

#ifndef INCLUDE_RELE_APPROXIMATORS_DATA_NORMALIZATION_H_
#define INCLUDE_RELE_APPROXIMATORS_DATA_NORMALIZATION_H_

#include "rele/approximators/data/BatchDataTraits.h"

#define FEATURES_TRAITS(dense) \
	using features_type = typename input_traits<dense>::column_type; \
	using collection_type = typename input_traits<dense>::type;

namespace ReLe
{

/*!
 * This interface is used to implement a normalization algorithm over a dataset.
 * \see ReLe::normalizeDataset
 * \see ReLe::normalizeDatasetFull
 */
template<bool dense = true>
class Normalization
{
    FEATURES_TRAITS(dense)
public:
    /*!
     * Normalization operator.
     * \param features the features vector to be normalized
     * \return the normalized features vector
     * \see normalize
     */
    inline features_type operator()(const features_type& features)
    {
        return normalize(features);
    }

    /*!
     *	Compute normalization of a single input feature.
     *	\param features the features vector to be normalized
     *  \return the normalized features vector
     */
    virtual features_type normalize(const features_type& features) const = 0;

    /*!
     * Inverse normalization operation over an input feature.
     * \param features the features vector to be restored
     * \return the restored features vector
     */
    virtual features_type restore(const features_type& features) const = 0;


    /*!
     * This method is similar to restore, but, differently from Normalization::restore, it does not
     * add the dataset mean, only the relative elements scales.
     * \return the rescaled features vector
     */
    virtual features_type rescale(const features_type& features) const = 0;

    /*!
     * Read the whole dataset in order to compute the parameters for the normalization algorithm.
     * \param dataset the dataset over which the normalization should be performed
     */
    virtual void readData(const collection_type& dataset) = 0;


    /*!
     * Destructor.
     */
    virtual ~Normalization()
    {
    }

};


/*!
 * This class implements a fake normaliztion algorithm.
 * This algorithm doesn't perform any normalization.
 */
template<bool dense = true>
class NoNormalization : public Normalization<dense>
{
    FEATURES_TRAITS(dense)
public:
    virtual features_type normalize(const features_type& features) const override
    {
        return features;
    }

    virtual features_type restore(const features_type& features) const override
    {
        return features;
    }

    virtual features_type rescale(const features_type& features) const override
    {
        return features;
    }

    virtual void readData(const collection_type& dataset) override
    {

    }

    /*!
     * Destructor.
     */
    virtual ~NoNormalization()
    {
    }

};

/*!
 * This class implements the normalization over an interval.
 * Simply rescales the input features into a new range.
 */
template<bool dense = true>
class MinMaxNormalization : public Normalization<dense>
{
    FEATURES_TRAITS(dense)
public:
    /*!
     * Constructor.
     * \param minValue the new minimum value of the dataset
     * \param maxValue the new maximum value for the dataset
     */
    MinMaxNormalization(double minValue = 0.0, double maxValue = 1.0)
        : minValue(minValue), maxValue(maxValue)
    {

    }

    virtual features_type normalize(const features_type& features) const override
    {
        return (features - min)%deltaFactor+newMin;
    }

    virtual features_type restore(const features_type& features) const override
    {
        return (features - newMin)/deltaFactor + min;
    }

    virtual features_type rescale(const features_type& features) const override
    {
        return features/deltaFactor;
    }

    virtual void readData(const collection_type& dataset) override
    {
        min = arma::min(dataset, 1);
        arma::vec delta = arma::max(dataset, 1) - min;

        newMin.ones(dataset.n_rows);
        newMin *= minValue;

        arma::vec newDelta(dataset.n_rows, arma::fill::ones);
        newDelta *= maxValue - minValue;

        deltaFactor = newDelta/delta;
    }

    /*!
     * Destructor.
     */
    virtual ~MinMaxNormalization()
    {

    }

private:
    arma::vec min;
    arma::vec newMin;
    arma::vec deltaFactor;

    const double minValue;
    const double maxValue;

};

/*!
 * This class implements the z-score normalization.
 * This type of normalization consisty of subtracting the dataset mean and
 * dividing each element of the features vector by it's standard deviation.
 */
template<bool dense = true>
class ZscoreNormalization : public Normalization<dense>
{
    FEATURES_TRAITS(dense)
public:
    ZscoreNormalization()
    {

    }

    virtual features_type normalize(const features_type& features) const override
    {
        return (features - mean)/stddev;
    }

    virtual features_type restore(const features_type& features) const override
    {
        return features%stddev + mean;
    }

    virtual features_type rescale(const features_type& features) const override
    {
        return features%stddev;
    }

    virtual void readData(const collection_type& dataset) override
    {
        mean = arma::mean(dataset, 1);
        stddev = arma::stddev(dataset, 0, 1);
    }



    virtual ~ZscoreNormalization()
    {

    }

private:
    arma::vec mean;
    arma::vec stddev;
};


}

#endif /* INCLUDE_RELE_APPROXIMATORS_DATA_NORMALIZATION_H_ */
