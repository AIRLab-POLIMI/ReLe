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

#ifndef INCLUDE_RELE_APPROXIMATORS_REGRESSORS_TREES_LEAFTREENODE_H_
#define INCLUDE_RELE_APPROXIMATORS_REGRESSORS_TREES_LEAFTREENODE_H_

#include "TreeNode.h"

namespace ReLe
{

enum LeafType
{
    Constant,
    Linear,
    Samples
};

template<class OutputC, bool denseOutput>
class LeafTreeNode : public TreeNode<OutputC>
{

public:

    LeafTreeNode()
    {

    }

    LeafTreeNode(const BatchData_<OutputC, denseOutput>& data)
    {
        fit(data);
    }

    virtual ~LeafTreeNode()
    {

    }

    virtual void fit(const BatchData_<OutputC, denseOutput>& data)
    {
        value = data.getMean();
        variance = data.getVariance();
    }

    virtual OutputC getValue(const arma::vec& input) override
    {
        return value;
    }

    virtual bool isLeaf() override
    {
        return true;
    }

    virtual void writeOnStream(std::ofstream& out) override
    {
        out << "L" << std::endl;
        out << value << std::endl;
        out << variance << std::endl;
    }

    virtual void readFromStream(std::ifstream& in) override
    {
        //TODO [SERIALIZATION] implement
    }

protected:
    OutputC value; // The value
    arma::mat variance; //The variance

};

template<class OutputC, bool denseOutput>
class SampleLeafTreeNode : public LeafTreeNode<OutputC, denseOutput>
{
public:
    SampleLeafTreeNode(BatchData_<OutputC, denseOutput>* dataSet)
        : LeafTreeNode<OutputC, denseOutput>(*dataSet), dataSet(dataSet)
    {

    }

    ~SampleLeafTreeNode()
    {
        delete dataSet;
    }

private:
    BatchData_<OutputC, denseOutput>* dataSet;
};

template<class OutputC, bool denseOutput>
class LinearLeafTreeNode : public LeafTreeNode<OutputC, denseOutput>
{
    //TODO [MINOR] implement linear node
};

}



#endif /* INCLUDE_RELE_APPROXIMATORS_REGRESSORS_TREES_LEAFTREENODE_H_ */
