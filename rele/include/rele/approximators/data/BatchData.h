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

#include "RandomGenerator.h"

#include <set>
#include <vector>
#include <cassert>
#include "data/BatchDataTraits.h"

#define BATCH_DATA_TYPES(OutputC, dense) \
	using typename BatchData_<OutputC, dense>::features_type; \
    using typename BatchData_<OutputC, dense>::FeaturesCollection; \
    using typename BatchData_<OutputC, dense>::OutputCollection;

namespace ReLe
{

template<class InputC, class OutputC>
class BatchDataRaw_
{
public:
    BatchDataRaw_()
    {

    }

    BatchDataRaw_(std::vector<InputC> inputs, std::vector<OutputC> outputs) :
        inputs(inputs), outputs(outputs)
    {
        assert(inputs.size() == outputs.size());
    }

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

    void addSample(const InputC& input, const OutputC& output)
    {
        inputs.push_back(input);
        outputs.push_back(output);
    }

    virtual InputC getInput(unsigned int index) const
    {
        return inputs[index];
    }

    virtual OutputC getOutput(unsigned int index) const
    {
        return outputs[index];
    }

    virtual size_t size() const
    {
        return inputs.size();
    }

    virtual ~BatchDataRaw_()
    {

    }

private:
    std::vector<InputC> inputs;
    std::vector<OutputC> outputs;

};

template<class OutputC, bool dense>
class MiniBatchData_;

template<class OutputC, bool dense = true>
class BatchData_
{

public:
    typedef typename input_traits<dense>::column_type features_type;
    typedef typename input_traits<dense>::type FeaturesCollection;
    typedef typename output_traits<OutputC>::type OutputCollection;

public:
    BatchData_()
    {
        computed = false;
    }

    virtual features_type getInput(unsigned int index) const = 0;
    virtual OutputC getOutput(unsigned int index) const = 0;
    virtual size_t size() const = 0;

    virtual size_t featuresSize() const = 0;
    virtual OutputCollection getOutputs() const = 0;
    virtual FeaturesCollection getFeatures() const = 0;


    virtual BatchData_* clone() const = 0;
    virtual BatchData_* cloneSubset(const arma::uvec& indexes) const = 0;

    virtual const BatchData_* shuffle() const
    {
        arma::uvec indexes = arma::linspace<arma::uvec>(0, size() - 1, size());
        indexes = arma::shuffle(indexes);

        return new MiniBatchData_<OutputC, dense>(this, indexes);
    }

    virtual const std::vector<MiniBatchData_<OutputC, dense>*> getMiniBatches(unsigned int mSize) const
    {
    	unsigned int nMiniBatches = size() / mSize + ((size() % mSize == 0) ? 0 : 1);

        return miniBatchesVector(nMiniBatches, mSize);
    }

    virtual const std::vector<MiniBatchData_<OutputC, dense>*> getNMiniBatches(double nMiniBatches) const
    {
    	unsigned int mSize = size() / nMiniBatches;
    	if(size() % 2 != 0)
    		nMiniBatches--;

    	return miniBatchesVector(nMiniBatches, mSize);
    }

    OutputC getMean() const
    {
        if (!computed)
        {
            computeMeanVariance();
            computed = true;
        }

        return mean;
    }

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
    	assert(mSize > 0 && nMiniBatches > 0);

    	std::vector<MiniBatchData_<OutputC, dense>*> miniBatches;

        arma::uvec indexes = arma::linspace<arma::uvec>(0, size() - 1, size());
        indexes = arma::shuffle(indexes);

        unsigned int end;
        for (unsigned int i = 0; i < nMiniBatches; i++)
        {
            unsigned int start = i * mSize;
            end = start + mSize - 1;

            auto* miniBatch = new MiniBatchData_<OutputC, dense>(this, indexes.rows(start, end));
            miniBatches.push_back(miniBatch);
        }

        if(end < size() - 1)
        {
        	auto* miniBatch = new MiniBatchData_<OutputC, dense>(
        		this, indexes.rows(nMiniBatches * mSize, size()));
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

typedef BatchData_<arma::vec> BatchData;

template<class OutputC, bool dense = true>
class MiniBatchData_: public BatchData_<OutputC, dense>
{
    BATCH_DATA_TYPES(OutputC, dense)

public:
    MiniBatchData_(const BatchData_<OutputC, dense>* data,
                   const arma::uvec& indexes) :
        data(*data), indexes(indexes)
    {

    }

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

    const arma::uvec getIndexes() const
    {
    	return indexes;
    }

    void setIndexes(const arma::uvec& indexes)
    {
        this->computed = false;
        this->indexes = indexes;
    }

    virtual ~MiniBatchData_()
    {

    }

private:
    const BatchData_<OutputC, dense>& data;
    arma::uvec indexes;
};

typedef MiniBatchData_<arma::vec> MiniBatchData;

template<class OutputC, bool dense = true>
class BatchDataSimple_: public BatchData_<OutputC, dense>
{
    BATCH_DATA_TYPES(OutputC, dense)

public:
    BatchDataSimple_(const FeaturesCollection& features, const OutputCollection& outputs) :
        features(features), outputs(outputs)
    {
        assert(features.n_cols == outputs.n_cols);
    }

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

    virtual ~BatchDataSimple_()
    {

    }

private:
    FeaturesCollection features;
    OutputCollection outputs;

};

typedef BatchDataSimple_<arma::vec> BatchDataSimple;

}

#endif /* INCLUDE_RELE_APPROXIMATORS_BATCHDATA_H_ */
