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

#ifndef INCLUDE_RELE_APPROXIMATORS_DATA_BATCHDATANORMALIZATION_H_
#define INCLUDE_RELE_APPROXIMATORS_DATA_BATCHDATANORMALIZATION_H_

#include "rele/approximators/data/BatchData.h"
#include "rele/approximators/data/Normalization.h"

namespace ReLe
{

/*!
 * This function can be used to create a normalized version of the dataset.
 * Only input features are normalized.
 * \param dataset the dataset to be normalized.
 * \param normalization the normalization object to be used
 * \param computeNormalization if the normalization object should be initialized
 * by computing the normalization parameters
 * \return the normalized dataset.
 */
template<class OutputC, bool dense>
BatchDataSimple_<OutputC, dense> normalizeDataset(
    const BatchData_<OutputC, dense>& dataset,
    Normalization<dense>& normalization, bool computeNormalization =
        false)
{
    auto&& features = dataset.getFeatures();

    if (computeNormalization)
        normalization.readData(features);

    for (unsigned int i = 0; i < features.n_cols; i++)
    {
        features.col(i) = normalization(features.col(i));
    }

    return BatchDataSimple_<OutputC, dense>(features, dataset.getOutputs());
}

/*!
 * This function can be used to create a normalized version of the dataset.
 * Both input features and output are normalized.
 * \param dataset the dataset to be normalized.
 * \param featuresNormalization the normalization object to be used for input features
 * \param outputNormalization the normalization object to be used for outputs
 * \param computeNormalization if the normalization object should be initialized
 * by computing the normalization parameters
 * \return the normalized dataset.
 */
template<bool dense>
BatchDataSimple_<arma::vec, dense> normalizeDatasetFull(
    const BatchData_<arma::vec, dense>& dataset,
    Normalization<dense>& featuresNormalization,
    Normalization<true>& outputNormalization,
    bool computeNormalization = false)
{

    auto&& features = dataset.getFeatures();
    auto&& outputs = dataset.getOutputs();

    if(computeNormalization)
    {
        featuresNormalization.readData(features);
        outputNormalization.readData(outputs);
    }

    for (unsigned int i = 0; i < features.n_cols; i++)
    {
        features.col(i) = featuresNormalization(features.col(i));
        outputs.col(i) = outputNormalization(outputs.col(i));
    }

    return BatchDataSimple_<arma::vec, dense>(features, outputs);
}

}

#endif /* INCLUDE_RELE_APPROXIMATORS_DATA_BATCHDATANORMALIZATION_H_ */
