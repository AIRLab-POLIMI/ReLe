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

#include "trees/nodes/TreeNode.h"

namespace ReLe
{

/**
 * LeafTreeNode is a template class that represents a leaf of a
 * regression tree.
 * This class extends rtANode and contains methods to set/get the value
 * saved in the node, this value is of type OutputC.
 */
template<class InputC, class OutputC>
class LeafTreeNode : public TreeNode<InputC, OutputC>
{
public:

    /**
     * Empty Constructor
     */
    LeafTreeNode()
    {

    }

    /**
     * Basic constructor
     * @param val the value to store in the node
     */
    LeafTreeNode(const BatchData<InputC, OutputC>& data)
    {

    }

    /**
     *
     */
    virtual ~LeafTreeNode();

    /**
     * Set the value
     * @param val the value
     */
    virtual void fit(const BatchData<InputC, OutputC>& data)
    {
        //TODO implement
    }

    /**
     * Get the value
     * @return the value
     */
    virtual OutputC getValue(InputC& input);

    /**
     * This method is used to determine if the object is a leaf or an
     * internal node
     * @return true if it is a leaf, false otherwise
     */
    virtual bool isLeaf()
    {
    	return true;
    }

    /**
     *
     */
    virtual void WriteOnStream(std::ofstream& out)
    {
        out << "L" << std::endl;
        out << mValue << std::endl;
        out << mVariance << std::endl;
    }

    /**
     *
     */
    virtual void ReadFromStream(std::ifstream& in)
    {
        //TODO implement
    }

protected:
    OutputC mValue; // The value
    OutputC mVariance;

};

}



#endif /* INCLUDE_RELE_APPROXIMATORS_REGRESSORS_TREES_LEAFTREENODE_H_ */