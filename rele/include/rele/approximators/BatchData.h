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

namespace ReLe
{

//Traits to handle output
template<class OutputC>
struct output_traits
{
    static arma::mat square(const OutputC& o)
    {
        return arma::mat();
    }

    static bool isAlmostEqual(const OutputC& o1, const OutputC& o2)
    {
        return o1 == o2;
    }

};

template<>
struct output_traits<arma::vec>
{
    static arma::mat square(const arma::vec& o)
    {
        return o*o.t();
    }

    static bool isAlmostEqual(const arma::vec& o1, const arma::vec& o2)
    {
        return arma::norm(o1 - o2) < 1e-7;
    }

};

template<>
struct output_traits<unsigned int>
{
    static arma::mat square(const unsigned int& o)
    {
        arma::mat m;
        m = o*o;
        return m;
    }

    static bool isAlmostEqual(const unsigned int& o1, const unsigned int& o2)
    {
        return std::abs(o1 - o2) < 1e-7;
    }
};


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

    virtual const InputC& getInput(unsigned int index) const = 0;
    virtual const OutputC& getOutput(unsigned int index) const = 0;
    virtual size_t size() const = 0;

    virtual BatchData* clone() const
    {
        BatchDataPlain<InputC, OutputC>* newDataset = new BatchDataPlain<InputC, OutputC>();

        for(int i = 0; i < size(); i++)
        {
            newDataset->addSample(getInput(i), getOutput(i));
        }

        return newDataset;
    }

    const BatchData* getMiniBatch(unsigned int mSize) const
    {
        if(mSize >= size())
        {
            return new MiniBatchData<InputC, OutputC>(this);
        }

        std::set<unsigned int> indexesSet;
        std::vector<unsigned int> indexes;

        while(indexes.size() != mSize)
        {
            unsigned int r;
            do
            {
                r = RandomGenerator::sampleUniformInt(0, size() - 1);
            }
            while(indexesSet.count(r) != 0);

            indexes.push_back(r);
            indexesSet.insert(r);
        }

        return new MiniBatchData<InputC, OutputC>(this, indexes);
    }

    OutputC getMean() const
    {
        if(!computed)
        {
            computeMeanVariance();
            computed = true;
        }

        return mean;
    }

    arma::mat getVariance() const
    {
        if(!computed)
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
class MiniBatchData : public BatchData<InputC, OutputC>
{
public:
    MiniBatchData(const BatchData<InputC, OutputC>* data, const std::vector<unsigned int>& indexes) :
        data(*data), indexes(indexes)
    {

    }

    MiniBatchData(const BatchData<InputC, OutputC>& data, const std::vector<unsigned int>& indexes) :
        data(data), indexes(indexes)
    {

    }

    MiniBatchData(const BatchData<InputC, OutputC>* data) :
        data(*data)

    {

    }

    MiniBatchData(const BatchData<InputC, OutputC>& data) :
        data(data)

    {

    }

    virtual const InputC& getInput(unsigned int index) const
    {
        if(indexes.size() == 0)
            return data.getInput(index);

        unsigned int realIndex = indexes[index];
        return data.getInput(realIndex);
    }

    virtual const OutputC& getOutput(unsigned int index) const
    {
        if(indexes.size() == 0)
            return data.getOutput(index);

        unsigned int realIndex = indexes[index];
        return data.getOutput(realIndex);
    }

    virtual size_t size() const
    {
        if(indexes.size() == 0)
            return data.size();

        return indexes.size();
    }

    void setIndexes(std::vector<unsigned int>& indexes)
    {
        this->computed = false;
        this->indexes = indexes;
    }

    virtual ~MiniBatchData()
    {

    }

private:
    const BatchData<InputC, OutputC>& data;
    std::vector<unsigned int> indexes;
};


template<class InputC, class OutputC>
class BatchDataPlain : public BatchData<InputC, OutputC>
{
public:
    BatchDataPlain()
    {

    }

    BatchDataPlain(std::vector<InputC> inputs,
                   std::vector<OutputC> outputs) : inputs(inputs), outputs(outputs)
    {
        assert(inputs.size() == outputs.size());
    }

    void addSample(const InputC& input, const OutputC& output)
    {
        inputs.push_back(input);
        outputs.push_back(output);
    }

    virtual const InputC& getInput(unsigned int index) const
    {
        return inputs[index];
    }

    virtual const OutputC& getOutput(unsigned int index) const
    {
        return outputs[index];
    }

    virtual size_t size() const
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






}



#endif /* INCLUDE_RELE_APPROXIMATORS_BATCHDATA_H_ */
