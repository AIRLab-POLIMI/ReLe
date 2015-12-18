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

#ifndef INCLUDE_RELE_APPROXIMATORS_REGRESSORS_TREES_INTERNALTREENODE_H_
#define INCLUDE_RELE_APPROXIMATORS_REGRESSORS_TREES_INTERNALTREENODE_H_

#include "TreeNode.h"

#include "data/BatchData.h"

namespace ReLe
{

/**
 * InternalTreeNode is a template class that represents an internal node
 * of a regression tree.
 * This class extends TreeNode and contains methods to set/get the index
 * used to split the tree, the split value and the pointers to the left
 * and right childs of the node (binary trees).
 * The splitting value is of type double.
 */
template<class OutputC>
class InternalTreeNode: public TreeNode<OutputC>
{
public:

    /**
     * Empty constructor
     */
    InternalTreeNode() :
        axis(-1), split(0), left(nullptr), right(nullptr)
    {
    }

    /**
     * Basic contructor
     * @param a the index of splitting
     * @param s the split value
     * @param l the pointer to left child
     * @param r the pointer to right child
     */
    InternalTreeNode(int a, double s, TreeNode<OutputC>* l,
                     TreeNode<OutputC>* r) :
        axis(a), split(s), left(l), right(r)
    {
    }

    /**
     * Get axis, axis is the index of the split
     * @return the axis
     */
    virtual int getAxis() override
    {
        return axis;
    }

    /**
     * Get Split
     * @return the split value
     */
    virtual double getSplit() override
    {
        return split;
    }

    /**
     * Get the value of the subtree
     * @return the value
     */
    virtual OutputC getValue(const arma::vec& input) override
    {
        if (input[axis] < split)
        {
            return left->getValue(input);
        }
        else
        {
            return right->getValue(input);
        }
    }

    /**
     * Get Left Child
     * @return a pointer to the left chid node
     */
    virtual TreeNode<OutputC>* getLeft() override
    {
        return left;
    }

    /**
     * Get Right Child
     * @return a pointer to the right child node
     */
    virtual TreeNode<OutputC>* getRight() override
    {
        return right;
    }

    /**
     * Set te axis
     * @param a the axis
     */
    void setAxis(int a)
    {
        axis = a;
    }

    /**
     * Set the split
     * @param s the split value
     */
    void setSplit(double s)
    {
        split = s;
    }

    /**
     * Set the left child
     * @param l a pointer to the left child node
     */
    void setLeft(TreeNode<OutputC>* l)
    {
        left = l;
    }

    /**
     * Set the right child     * @param r a pointer to the right child node
     */
    void setRight(TreeNode<OutputC>* r)
    {
        right = r;
    }

    /**
     * Empty destructor
     */
    virtual ~InternalTreeNode()
    {
        if (left != nullptr && !left->isEmpty())
        {
            delete left;
        }
        if (right != nullptr && !right->isEmpty())
        {
            delete right;
        }
    }

    /**
     *
     */
    virtual void writeOnStream(std::ofstream& out) override
    {
        out << "N" << std::endl;
        out << axis << " " << split;
        out << std::endl;
        if (left)
        {
            out << *left;
        }
        else
        {
            out << "Empty" << std::endl;
        }
        if (right)
        {
            out << *right;
        }
        else
        {
            out << "Empty" << std::endl;
        }
    }

    /**
     *
     */
    virtual void readFromStream(std::ifstream& in) override
    {
        //TODO implement
    }

private:
    int axis;  // the axis of split
    double split;  // the value of split
    TreeNode<OutputC>* left;  // pointer to right child
    TreeNode<OutputC>* right;  // pointer to left child
};

}

#endif /* INCLUDE_RELE_APPROXIMATORS_REGRESSORS_TREES_INTERNALTREENODE_H_ */
