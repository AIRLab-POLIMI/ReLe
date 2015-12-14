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

#include "BatchDataTraits.h"

#include <set>
#include <vector>
#include <cassert>

namespace ReLe
{

template<class InputC, class OutputC>
class MiniBatchData;

template<class InputC, class OutputC>
class BatchDataPlain;

template<class InputC, class OutputC>
class BatchData
{

public:
    BatchData()
    {
        computed = false;
    }

    virtual InputC getInput(unsigned int index) const = 0;
    virtual OutputC getOutput(unsigned int index) const = 0;
    virtual size_t size() const = 0;
    virtual BatchData* clone() const = 0;
    virtual BatchData* cloneSubset(const arma::uvec& indexes) const = 0;

    virtual const BatchData* shuffle() const
    {
        arma::uvec indexes = arma::linspace<arma::uvec>(0, size() - 1, size());
        indexes = arma::shuffle(indexes);

        return new MiniBatchData<InputC, OutputC>(this, indexes);
    }

    virtual std::vector<const BatchData*> getMiniBatches(unsigned int mSize) const
    {
        std::vector<const BatchData*> minibatches;

        arma::uvec indexes = arma::linspace<arma::uvec>(0, size() - 1, size());
        indexes = arma::shuffle(indexes);

        const unsigned int mNumber = indexes.n_elem / mSize + (indexes.n_elem % mSize == 0) ? 0 : 1;

        for (unsigned int i = 0; i < mNumber; i++)
        {
            unsigned int start = i * mSize;
            unsigned int last = indexes.n_elem;
            unsigned int end = std::min(start + mSize, last) -1;

            auto* miniBatch = new MiniBatchData<InputC, OutputC>(this, indexes.rows(start, end));

            minibatches.push_back(miniBatch);
        }

        return minibatches;
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

    virtual ~BatchData()
    {

    }

protected:
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

template<class InputC, class OutputC>
class MiniBatchData: public BatchData<InputC, OutputC>
{
public:
    MiniBatchData(const BatchData<InputC, OutputC>* data,
                  const arma::uvec& indexes) :
        data(*data), indexes(indexes)
    {

    }

    MiniBatchData(const BatchData<InputC, OutputC>& data,
                  const arma::uvec& indexes) :
        data(data), indexes(indexes)
    {

    }

    virtual BatchData<InputC, OutputC>* clone() const override
    {
        return new MiniBatchData<InputC, OutputC>(data, indexes);
    }

    virtual BatchData<InputC, OutputC>* cloneSubset(
        const arma::uvec& indexes) const override
    {
        arma::uvec newIndexes = this->indexes(indexes);
        return new MiniBatchData<InputC, OutputC>(data, newIndexes);
    }

    virtual const BatchData<InputC, OutputC>* shuffle() const override
    {
        if (data.size() == indexes.n_elem)
            return data.shuffle();
        else
            return BatchData<InputC, OutputC>::shuffle();
    }

    virtual std::vector<const BatchData<InputC, OutputC>*> getMiniBatches(unsigned int mSize) const override
    {
        if (data.size() == indexes.n_elem)
            return data.getMiniBatches(mSize);
        else
            return BatchData<InputC, OutputC>::getMiniBatches(mSize);
    }

    virtual InputC getInput(unsigned int index) const override
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

    void setIndexes(const arma::uvec& indexes)
    {
        this->computed = false;
        this->indexes = indexes;
    }

    virtual ~MiniBatchData()
    {

    }

private:
    const BatchData<InputC, OutputC>& data;
    arma::uvec indexes;
};

template<class InputC, class OutputC>
class BatchDataPlain: public BatchData<InputC, OutputC>
{
public:
    BatchDataPlain()
    {

    }

    BatchDataPlain(std::vector<InputC> inputs, std::vector<OutputC> outputs) :
        inputs(inputs), outputs(outputs)
    {
        assert(inputs.size() == outputs.size());
    }

    virtual BatchData<InputC, OutputC>* clone() const override
    {
        BatchDataPlain<InputC, OutputC>* newDataset = new BatchDataPlain<InputC,
        OutputC>();

        for (int i = 0; i < size(); i++)
        {
            newDataset->addSample(getInput(i), getOutput(i));
        }

        return newDataset;
    }

    virtual BatchData<InputC, OutputC>* cloneSubset(
        const arma::uvec& indexes) const override
    {
        return new MiniBatchData<InputC, OutputC>(this, indexes);
    }

    void addSample(const InputC& input, const OutputC& output)
    {
        inputs.push_back(input);
        outputs.push_back(output);
    }

    virtual InputC getInput(unsigned int index) const override
    {
        return inputs[index];
    }

    virtual OutputC getOutput(unsigned int index) const override
    {
        return outputs[index];
    }

    virtual size_t size() const override
    {
        return inputs.size();
    }

    virtual ~BatchDataPlain()
    {

    }

private:
    std::vector<InputC> inputs;
    std::vector<OutputC> outputs;

};

template<class InputC, class OutputC>
class BatchDataFeatures: public BatchData<InputC, OutputC>
{
public:
    typedef typename input_collection<InputC>::const_ref_type FeaturesCollection;
    typedef typename output_collection<OutputC>::const_ref_type OutputCollection;

public:
    BatchDataFeatures(FeaturesCollection features, OutputCollection outputs) :
        features(features), outputs(outputs)
    {
        assert(features.n_cols == outputs.n_cols);
    }

    virtual size_t size() const override
    {
        return features.n_cols;
    }

    virtual size_t featuresSize() const
    {
        return features.n_rows;
    }

    arma::vec getMinFeatures() const
    {
        return arma::min(features, 1);
    }

    arma::vec getMaxFeatures() const
    {
        return arma::max(features, 1);
    }

    arma::vec getMeanFeatures() const
    {
        return arma::mean(features, 1);
    }

    arma::vec getStdDevFeatures() const
    {
        return arma::stddev(features, 0, 1);
    }

    virtual BatchData<InputC, OutputC>* clone() const override
    {
        return new BatchDataFeatures<InputC, OutputC>(features, outputs);
    }

    virtual BatchData<InputC, OutputC>* cloneSubset(
        const arma::uvec& indexes) const override
    {
        return new MiniBatchData<InputC, OutputC>(this, indexes);
    }

    virtual InputC getInput(unsigned int index) const override
    {
        return features.col(index);
    }

    virtual OutputC getOutput(unsigned int index) const override
    {
        return outputs.col(index);
    }

    FeaturesCollection getOutputs()
    {
        return outputs;
    }

    OutputCollection getFeatures()
    {
        return features;
    }

    virtual ~BatchDataFeatures()
    {

    }

private:
    FeaturesCollection features;
    OutputCollection outputs;

};

}

#endif /* INCLUDE_RELE_APPROXIMATORS_BATCHDATA_H_ */
