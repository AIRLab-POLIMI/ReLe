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

#ifndef INCLUDE_RELE_APPROXIMATORS_REGRESSORS_TREES_NODES_EMPTYTREENODE_H_
#define INCLUDE_RELE_APPROXIMATORS_REGRESSORS_TREES_NODES_EMPTYTREENODE_H_

namespace ReLe
{

template<class OutputC>
class EmptyTreeNode : public TreeNode<OutputC>
{
public:
    EmptyTreeNode(const OutputC& defaultValue) : defaultValue(defaultValue)
    {

    }

    /**
     * Get the value
     * @return the value
     */
    virtual OutputC getValue(const arma::vec& input)
    {
        return defaultValue;
    }

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
     * This method is used to determine if the object is an empty node leaf or not
     * @return true if it is an empty leaf, false otherwise
     */
    virtual bool isEmpty()
    {
        return true;
    }

    /**
     *
     */
    virtual void writeOnStream (std::ofstream& out)
    {
        out << "EmptyNode" << std::endl;
    }

    /**
     *
     */
    virtual void readFromStream (std::ifstream& in)
    {

    }

    virtual ~EmptyTreeNode()
    {

    }

private:
    OutputC defaultValue;

};

}


#endif /* INCLUDE_RELE_APPROXIMATORS_REGRESSORS_TREES_NODES_EMPTYTREENODE_H_ */
