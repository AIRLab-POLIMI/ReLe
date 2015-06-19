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

#ifndef INCLUDE_RELE_APPROXIMATORS_REGRESSORS_TREES_TREENODE_H_
#define INCLUDE_RELE_APPROXIMATORS_REGRESSORS_TREES_TREENODE_H_

#include <iostream>

namespace ReLe
{

/**
 * AbstractTreeNode is a class that represents an abstract node of a regression
 * tree. The method isLeaf() is used to determine if it is a leaf or an
 * internal node.
 */
template<class InputC, class OutputC>
class TreeNode
{
public:

    /**
     * Empty Constructor
     */
    TreeNode() {}

    /**
     * Empty Destructor
     */
    virtual ~TreeNode() {}

    /**
     * This method is used to determine if the object is a leaf or an
     * internal node
     * @return true if it is a leaf, false otherwise
     */
    virtual bool isLeaf()
    {
        return false;
    }

    /**
     * Get axis, axis is the index of the split
     * @return the axis
     */
    virtual int getAxis()
    {
        return -1;
    }

    /**
     * Get Split
     * @return the split value
     */
    //FIXME wrong doc
    virtual OutputC getValue(InputC& input)
    {
        return -1;
    }

    /**
     * Get Split
     * @return the split value
     */
    virtual double getSplit()
    {
        return -1;
    }

    /**
     * Get Left Child
     * @return a pointer to the left child node
     */
    virtual TreeNode<InputC, OutputC>* getLeft()
    {
        return nullptr;
    }

    /**
     * Get Right Child
     * @return a pointer to the right child node
     */
    virtual TreeNode<InputC, OutputC>* getRight()
    {
        return nullptr;
    }

    /**
     * Get Child
     * @return a pointer to the right child node
     */
    virtual TreeNode<InputC, OutputC>* getChild(double* values = 0)
    {
        if (values == 0)
            return nullptr;
        return nullptr;
    }

    /**
     *
     */
    virtual void writeOnStream (std::ofstream& out) = 0;

    /**
     *
     */
    virtual void readFromStream (std::ifstream& in) = 0;

    /**
     *
     */
    friend std::ofstream& operator<< (std::ofstream& out, TreeNode<InputC, OutputC>& n)
    {
        n.writeOnStream(out);
        return out;
    }

    /**
     *
     */
    friend std::ifstream& operator>> (std::ifstream& in, TreeNode<InputC, OutputC>& n)
    {
        n.readFromStream(in);
        return in;
    }

};


}

#endif /* INCLUDE_RELE_APPROXIMATORS_REGRESSORS_TREES_TREENODE_H_ */