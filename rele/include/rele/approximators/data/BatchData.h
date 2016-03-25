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

#ifndef INCLUDE_RELE_APPROXIMATORS_BATCHDATA_H_
#define INCLUDE_RELE_APPROXIMATORS_BATCHDATA_H_

#include "rele/utils/RandomGenerator.h"
#include "rele/approximators/data/BatchDataTraits.h"

#include <set>
#include <vector>
#include <cassert>

#define BATCH_DATA_TYPES(OutputC, dense) \
	using typename BatchData_<OutputC, dense>::features_type; \
    using typename BatchData_<OutputC, dense>::FeaturesCollection; \
    using typename BatchData_<OutputC, dense>::OutputCollection;

namespace ReLe
{
/*!
 * This class represents a dataset of raw input/output data, that can be used to
 * train a ReLe::BatchRegressor
 */
template<class InputC, class OutputC>
class BatchDataRaw_
{
public:
    /*!
     * Constructor.
     */
    BatchDataRaw_()
    {

    }

    /*!
     * Constructor.
     * \param inputs the vector of input data
     * \param outputs the vector of output data, corresponding to the inputs
     */
    BatchDataRaw_(std::vector<InputC> inputs, std::vector<OutputC> outputs) :
        inputs(inputs), outputs(outputs)
    {
        assert(inputs.size() == outputs.size());
    }

    /*!
     * Create a copy of this object, containing an exact copy of the dataset.
     * \return a pointer to the copy
     */
    virtual BatchDataRaw_<InputC, OutputC>* clone() const
    {
        BatchDataRaw_<InputC, OutputC>* newDataset = new BatchDataRaw_<InputC,
        OutputC>();

        for (int i = 0; i < size(); i++)
        {
            newDataset->addSample(getInput(i), getOutput(i));
        }

        return newDataset;
    }

    /*!
     * Add a sample to the dataset.
     * \param input the input to be added
     * \param output the corresponding output
     */
    void addSample(const InputC& input, const OutputC& output)
    {
        inputs.push_back(input);
        outputs.push_back(output);
    }

    /*!
     * Getter.
     * \param index the index of the input in the dataset
     * \return the corresponding input
     */
    virtual InputC getInput(unsigned int index) const
    {
        return inputs[index];
    }

    /*!
     * Getter.
     * \param index the index of the output in the dataset
     * \return the corresponding output
     */
    virtual OutputC getOutput(unsigned int index) const
    {
        return outputs[index];
    }

    /*!
     * Getter.
     * \return the number of input output couples in the dataset.
     */
    virtual size_t size() const
    {
        return inputs.size();
    }

    /*!
     * Destructor.
     */
    virtual ~BatchDataRaw_()
    {

    }

private:
    std::vector<InputC> inputs;
    std::vector<OutputC> outputs;

};

template<class OutputC, bool dense>
class MiniBatchData_;


/*!
 * This interface represents a dataset of input/output data, where the input is a set of
 * precomputed features from the input dataset.
 * An implementation of this interface can be used to train a ReLe::BatchRegressor using
 * the low level method BatchRegressor::trainFeatures
 */
template<class OutputC, bool dense = true>
class BatchData_
{

public:
    //! the type of the input features
    typedef typename input_traits<dense>::column_type features_type;
    //! the type of the set of input features
    typedef typename input_traits<dense>::type FeaturesCollection;
    //! the type of the set of the outputs
    typedef typename output_traits<OutputC>::type OutputCollection;

public:
    /*!
     * Constructor.
     */
    BatchData_()
    {
        computed = false;
    }

    /*!
     * Getter.
     * \param index the index of the input feature vector in the dataset
     * \return the corresponding input feature vector
     */
    virtual features_type getInput(unsigned int index) const = 0;

    /*!
     * Getter.
     * \param index the index of the output in the dataset
     * \return the corresponding output
     */
    virtual OutputC getOutput(unsigned int index) const = 0;
    /*!
     * Getter.
     * \return the number of input output couples in the dataset.
     */
    virtual size_t size() const = 0;

    /*!
     * Getter.
     * \return the dimensionality of the features vectors
     */
    virtual size_t featuresSize() const = 0;

    /*!
     * Getter.
     * \return the output collection
     */
    virtual OutputCollection getOutputs() const = 0;

    /*!
     * Getter.
     * \return the input features.
     */
    virtual FeaturesCollection getFeatures() const = 0;

    /*!
     * Create a copy of this object, containing an exact copy of the dataset.
     * \return a pointer to the copy
     */
    virtual BatchData_* clone() const = 0;

    /*!
     * Create a copy of a subste of the dataset.
     * \param indexes the set of input features/outputs to be copied
     * \return a pointer to the copy of the subset of the dataset
     */
    virtual BatchData_* cloneSubset(const arma::uvec& indexes) const = 0;

    /*!
     * Creates a shuffled copy of the dataset.
     * \return a ReLe::MiniBatchData_ object with shuffled indexes (by default)
     */
    virtual const BatchData_* shuffle() const
    {
        arma::uvec indexes = arma::linspace<arma::uvec>(0, size() - 1, size());
        indexes = arma::shuffle(indexes);

        return new MiniBatchData_<OutputC, dense>(this, indexes);
    }

    /*!
     * Split the dataset in minibatches of constant size.
     * \param minibatchSize the size of all minibatches (except last)
     * \return a vector containing a set of pointers to the minibatches
     */
    virtual const std::vector<MiniBatchData_<OutputC, dense>*> getMiniBatches(unsigned int minibatchSize) const
    {
        assert(minibatchSize > 0);

        // Number of minibatches of the same size
        unsigned int nMiniBatches = size() / minibatchSize;

        return miniBatchesVector(nMiniBatches, minibatchSize);
    }

    /*!
     * Split the dataset in a set of N minibatches of equal length.
     * The last one might have a different size.
     * \param nMiniBatches the number of minibatches to use
     * \return a vector containing a set of pointers to the minibatches
     */
    virtual const std::vector<MiniBatchData_<OutputC, dense>*> getNMiniBatches(unsigned int nMiniBatches) const
    {
        assert(nMiniBatches > 0);

        unsigned int mSize = size() / nMiniBatches;
        nMiniBatches--;

        return miniBatchesVector(nMiniBatches, mSize);
    }

    /*!
     * Getter.
     * \return the mean output value of the dataset.
     */
    OutputC getMean() const
    {
        if (!computed)
        {
            computeMeanVariance();
            computed = true;
        }

        return mean;
    }

    /*!
     * Getter.
     * \return the output variance of the dataset.
     */
    arma::mat getVariance() const
    {
        if (!computed)
        {
            computeMeanVariance();
            computed = true;
        }

        return variance;
    }

    virtual ~BatchData_()
    {

    }

protected:
    virtual const std::vector<MiniBatchData_<OutputC, dense>*> miniBatchesVector(unsigned int nMiniBatches, unsigned int mSize) const
    {
        assert(mSize > 0);

        std::vector<MiniBatchData_<OutputC, dense>*> miniBatches;

        arma::uvec indexes = arma::linspace<arma::uvec>(0, size() - 1, size());
        indexes = arma::shuffle(indexes);

        unsigned int end = 0;
        for (unsigned int i = 0; i < nMiniBatches; i++)
        {
            unsigned int start = i * mSize;
            end = start + mSize;

            auto* miniBatch = new MiniBatchData_<OutputC, dense>(this, indexes.rows(start, end - 1));
            miniBatches.push_back(miniBatch);
        }

        if(end < size())
        {
            auto* miniBatch = new MiniBatchData_<OutputC, dense>(
                this, indexes.rows(end * nMiniBatches, size() - 1));
            miniBatches.push_back(miniBatch);
        }

        return miniBatches;
    }

    void computeMeanVariance() const
    {
        if (size() == 0)
        {
            return;
        }

        const OutputC& output = getOutput(0);
        mean = output;
        arma::mat m2 = output_traits<OutputC>::square(output);

        for (int i = 1; i < size(); i++)
        {
            const OutputC& output = getOutput(i);
            mean += output;
            m2 += output_traits<OutputC>::square(output);
        }

        mean /= size();
        m2 /= size();
        variance = m2 - output_traits<OutputC>::square(mean);
    }

protected:
    mutable bool computed;

private:
    mutable arma::mat variance;
    mutable OutputC mean;
};

//! Template alias.
typedef BatchData_<arma::vec> BatchData;


/*!
 * Implementation of a minibatch of a dataset.
 */
template<class OutputC, bool dense = true>
class MiniBatchData_: public BatchData_<OutputC, dense>
{
    BATCH_DATA_TYPES(OutputC, dense)

public:
    /*!
     * Construtcor.
     * \param data the original dataset.
     * \param indexes the set of elements of the original dataset.
     */
    MiniBatchData_(const BatchData_<OutputC, dense>* data,
                   const arma::uvec& indexes) :
        data(*data), indexes(indexes)
    {

    }

    /*!
     * Construtcor.
     * \param data the original dataset.
     * \param indexes the set of elements of the original dataset.
     */
    MiniBatchData_(const BatchData_<OutputC, dense>& data,
                   const arma::uvec& indexes) :
        data(data), indexes(indexes)
    {

    }

    virtual BatchData_<OutputC, dense>* clone() const override
    {
        return new MiniBatchData_<OutputC, dense>(data, indexes);
    }

    virtual BatchData_<OutputC, dense>* cloneSubset(
        const arma::uvec& indexes) const override
    {
        arma::uvec newIndexes = this->indexes(indexes);
        return new MiniBatchData_<OutputC, dense>(data, newIndexes);
    }

    virtual const BatchData_<OutputC, dense>* shuffle() const override
    {
        if (data.size() == indexes.n_elem)
            return data.shuffle();
        else
            return BatchData_<OutputC, dense>::shuffle();
    }

    virtual const std::vector<MiniBatchData_<OutputC, dense>*> getMiniBatches(unsigned int mSize) const override
    {
        if (data.size() == indexes.n_elem)
            return data.getMiniBatches(mSize);
        else
            return MiniBatchData_<OutputC, dense>::getMiniBatches(mSize);
    }

    virtual features_type getInput(unsigned int index) const override
    {
        unsigned int realIndex = indexes(index);
        return data.getInput(realIndex);
    }

    virtual OutputC getOutput(unsigned int index) const override
    {
        unsigned int realIndex = indexes(index);
        return data.getOutput(realIndex);
    }

    virtual size_t size() const override
    {
        return indexes.n_elem;
    }

    virtual size_t featuresSize() const override
    {
        return data.featuresSize();
    }

    virtual OutputCollection getOutputs() const override
    {
        return data.getOutputs().cols(indexes);
    }


    virtual FeaturesCollection getFeatures() const override
    {
        return data.getFeatures().cols(indexes);
    }

    /*!
     * Getter.
     * \return the indexes of the elements of the original dataset used by the minibatch
     */
    const arma::uvec getIndexes() const
    {
        return indexes;
    }

    /*!
     * Setter.
     * \param indexes the indexes of the element of the original dataset to be used
     */
    void setIndexes(const arma::uvec& indexes)
    {
        this->computed = false;
        this->indexes = indexes;
    }

    /*!
     * Destructor.
     */
    virtual ~MiniBatchData_()
    {

    }

public:
    /*!
     * Method used to cleanup a vector of pointers to minibatches
     * \param miniBatches the vector of pointers to clean
     */
    static void cleanMiniBatches(std::vector<MiniBatchData_<OutputC, dense>*> miniBatches)
    {
        for(auto* mb : miniBatches)
            delete mb;
    }

private:
    const BatchData_<OutputC, dense>& data;
    arma::uvec indexes;
};

//! Template alias.
typedef MiniBatchData_<arma::vec> MiniBatchData;

/*!
 * Simple implementation of the ReLe::BatchData_ interface.
 * Stores all input data in the memory.
 */
template<class OutputC, bool dense = true>
class BatchDataSimple_: public BatchData_<OutputC, dense>
{
    BATCH_DATA_TYPES(OutputC, dense)

public:
    /*!
     * Constructor.
     * \param features the collection of input features
     * \param outputs the collection of the corresponding outputs
     */
    BatchDataSimple_(const FeaturesCollection& features, const OutputCollection& outputs) :
        features(features), outputs(outputs)
    {
        assert(features.n_cols == outputs.n_cols);
    }

    /*!
     * Constructor.
     * \param features the collection of input features
     * \param outputs the collection of the corresponding outputs
     */
    BatchDataSimple_(FeaturesCollection&& features, OutputCollection&& outputs) :
        features(features), outputs(outputs)
    {
        assert(features.n_cols == outputs.n_cols);
    }

    virtual size_t size() const override
    {
        return features.n_cols;
    }

    virtual size_t featuresSize() const override
    {
        return features.n_rows;
    }

    virtual BatchData_<OutputC, dense>* clone() const override
    {
        return new BatchDataSimple_<OutputC, dense>(features, outputs);
    }

    virtual BatchData_<OutputC, dense>* cloneSubset(
        const arma::uvec& indexes) const override
    {
        return new MiniBatchData_<OutputC, dense>(this, indexes);
    }

    virtual features_type getInput(unsigned int index) const override
    {
        return features.col(index);
    }

    virtual OutputC getOutput(unsigned int index) const override
    {
        return outputs.col(index);
    }

    virtual OutputCollection getOutputs() const override
    {
        return outputs;
    }

    virtual FeaturesCollection getFeatures() const override
    {
        return features;
    }

    /*!
     * Destructor.
     */
    virtual ~BatchDataSimple_()
    {

    }

private:
    FeaturesCollection features;
    OutputCollection outputs;

};

//! Template alias.
typedef BatchDataSimple_<arma::vec> BatchDataSimple;

}

#endif /* INCLUDE_RELE_APPROXIMATORS_BATCHDATA_H_ */
