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

#include "rele/approximators/data/BatchData.h"

namespace ReLe
{

template<class OutputC>
class InternalTreeNode: public TreeNode<OutputC>
{
public:

    InternalTreeNode() :
        axis(-1), split(0), left(nullptr), right(nullptr)
    {
    }

    InternalTreeNode(int a, double s, TreeNode<OutputC>* l,
                     TreeNode<OutputC>* r) :
        axis(a), split(s), left(l), right(r)
    {
    }

    virtual int getAxis() override
    {
        return axis;
    }

    virtual double getSplit() override
    {
        return split;
    }

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

    virtual TreeNode<OutputC>* getLeft() override
    {
        return left;
    }

    virtual TreeNode<OutputC>* getRight() override
    {
        return right;
    }

    void setAxis(int a)
    {
        axis = a;
    }

    void setSplit(double s)
    {
        split = s;
    }

    void setLeft(TreeNode<OutputC>* l)
    {
        left = l;
    }

    void setRight(TreeNode<OutputC>* r)
    {
        right = r;
    }

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

    virtual void readFromStream(std::ifstream& in) override
    {
        //TODO [SERIALIZATION] implement
    }

private:
    int axis;  // the axis of split
    double split;  // the value of split
    TreeNode<OutputC>* left;  // pointer to right child
    TreeNode<OutputC>* right;  // pointer to left child
};

}

#endif /* INCLUDE_RELE_APPROXIMATORS_REGRESSORS_TREES_INTERNALTREENODE_H_ */
