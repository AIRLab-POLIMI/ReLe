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

#include "data/BatchDataTraits.h"

#define FEATURES_TRAITS(dense) \
	using features_type = typename input_traits<dense>::column_type; \
	using collection_type = typename input_traits<dense>::type;

namespace ReLe
{

template<bool dense = true>
class Normalization
{
    FEATURES_TRAITS(dense)
public:
    inline features_type operator()(const features_type& features)
    {
        return normalize(features);
    }

    virtual features_type normalize(const features_type& features) const = 0;
    virtual void readData(const collection_type& dataset) = 0;


    virtual ~Normalization()
    {
    }

};



template<bool dense = true>
class NoNormalization : public Normalization<dense>
{
    FEATURES_TRAITS(dense)
public:
    virtual features_type normalize(const features_type& features) const override
    {
        return features;
    }

    virtual void readData(const collection_type& dataset) override
    {

    }

    virtual ~NoNormalization()
    {
    }

};

template<bool dense = true>
class MinMaxNormalization : public Normalization<dense>
{
    FEATURES_TRAITS(dense)
public:
    MinMaxNormalization(double minValue = 0.0, double maxValue = 1.0)
        : minValue(minValue), maxValue(maxValue)
    {

    }

    virtual void readData(const collection_type& dataset) override
    {
        min = arma::min(dataset, 1);
        arma::vec delta = arma::max(dataset, 1);
        - min;

        newMin.ones(dataset.n_rows);
        newMin *= minValue;

        arma::vec newDelta(dataset.n_rows, arma::fill::ones);
        newDelta *= maxValue - minValue;

        deltaFactor = newDelta/delta;
    }

    virtual features_type normalize(const features_type& features) const override
    {
        return (features - min)/deltaFactor+newMin;
    }

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


template<bool dense = true>
class ZscoreNormalization_ : public Normalization<dense>
{
    FEATURES_TRAITS(dense)
public:
    ZscoreNormalization_()
    {

    }

    virtual void readData(const collection_type& dataset) override
    {
        mean = arma::mean(dataset, 1);
        stddev = arma::stddev(dataset, 0, 1);
    }

    virtual features_type normalize(const features_type& features) const override
    {
        return (features - mean)/stddev;
    }

    virtual ~ZscoreNormalization_()
    {

    }

private:
    arma::vec mean;
    arma::vec stddev;
};


}

#endif /* INCLUDE_RELE_APPROXIMATORS_DATA_NORMALIZATION_H_ */
