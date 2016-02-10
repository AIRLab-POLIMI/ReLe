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

#ifndef INCLUDE_RELE_APPROXIMATORS_REGRESSORS_TREES_BITS_NODES_EMPTYTREENODE_H_
#define INCLUDE_RELE_APPROXIMATORS_REGRESSORS_TREES_BITS_NODES_EMPTYTREENODE_H_

#include "TreeNode.h"

namespace ReLe
{

template<class OutputC>
class EmptyTreeNode : public TreeNode<OutputC>
{
public:
    EmptyTreeNode(const OutputC& defaultValue) : defaultValue(defaultValue)
    {

    }

    virtual OutputC getValue(const arma::vec& input) override
    {
        return defaultValue;
    }

    virtual bool isLeaf() override
    {
        return true;
    }


    virtual bool isEmpty() override
    {
        return true;
    }

    virtual void writeOnStream (std::ofstream& out) override
    {
        out << "EmptyNode" << std::endl;
    }

    virtual void readFromStream (std::ifstream& in) override
    {

    }

    virtual ~EmptyTreeNode()
    {

    }

private:
    OutputC defaultValue;

};

}


#endif /* INCLUDE_RELE_APPROXIMATORS_REGRESSORS_TREES_BITS_NODES_EMPTYTREENODE_H_ */
