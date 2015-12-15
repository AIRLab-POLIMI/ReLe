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

#ifndef INCLUDE_RELE_APPROXIMATORS_REGRESSORS_NN_NN_BITS_NORMALIZATION_H_
#define INCLUDE_RELE_APPROXIMATORS_REGRESSORS_NN_NN_BITS_NORMALIZATION_H_


namespace ReLe
{


class Normalization
{
public:
    virtual arma::vec normalizeInput(const arma::vec& features) = 0;
    //virtual arma::vec normalizeOutput(const arma::vec& features) = 0;

    virtual ~Normalization()
    {
    }

};

class NoNormalization : public Normalization
{
public:
    virtual arma::vec normalizeInput(const arma::vec& features) override
    {
        return features;
    }

    virtual ~NoNormalization()
    {
    }

};

class MinMaxNormalization : public Normalization
{
public:
    MinMaxNormalization(const BatchDataFeatures& data, double minValue = 0.0, double maxValue = 1.0)
    {
        min = data.getMinFeatures();
        arma::vec delta = data.getMaxFeatures() - min;

        size_t featuresSize = data.featuresSize();

        newMin.ones(featuresSize);
        newMin *= minValue;

        arma::vec newDelta(featuresSize, arma::fill::ones);
        newDelta *= maxValue - minValue;

        deltaFactor = newDelta/delta;
    }

    virtual arma::vec normalizeInput(const arma::vec& features) override
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

};

class ZscoreNormalization : public Normalization
{
public:
    ZscoreNormalization(const BatchDataFeatures& data)
    {
        mean = data.getMeanFeatures();
        stddev = data.getStdDevFeatures();
    }

    virtual arma::vec normalizeInput(const arma::vec& features) override
    {
        return (features - mean)/stddev;
    }

    virtual ~ZscoreNormalization()
    {

    }

private:
    arma::vec mean;
    arma::vec stddev;
};


}

#endif /* INCLUDE_RELE_APPROXIMATORS_REGRESSORS_NN_NN_BITS_NORMALIZATION_H_ */
