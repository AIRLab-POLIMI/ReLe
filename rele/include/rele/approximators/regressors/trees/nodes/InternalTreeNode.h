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

#include "trees/nodes/TreeNode.h"

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
template<class InputC, class OutputC>
class InternalTreeNode : public TreeNode<InputC, OutputC>
{
public:

    /**
     * Empty constructor
     */
    InternalTreeNode() : axis(-1), split(0), left(nullptr), right(nullptr)
    {}

    /**
     * Basic contructor
     * @param a the index of splitting
     * @param s the split value
     * @param l the pointer to left child
     * @param r the pointer to right child
     */
    InternalTreeNode(int a, double s, TreeNode<InputC, OutputC>* l, TreeNode<InputC, OutputC>* r)
        : axis(a ), split(s), left(l), right(r)
    {}

    /**
     * Get axis, axis is the index of the split
     * @return the axis
     */
    virtual int getAxis()
    {
        return axis;
    }

    /**
     * Get Split
     * @return the split value
     */
    virtual double getSplit()
    {
        return split;
    }

    /**
     * Get Left Child
     * @return a pointer to the left chid node
     */
    virtual TreeNode<InputC, OutputC>* getLeft()
    {
        return left;
    }

    /**
      * Get Right Child
      * @return a pointer to the right child node
      */
    virtual TreeNode<InputC,OutputC>* getRight()
    {
        return right;
    }

    /**
     * Get Child
     * @return a pointer to the right child node
     */
    virtual TreeNode<InputC, OutputC>* getChild(double* values)
    {
        if (values[axis] < split)
        {
            return left;
        }
        else
        {
            return right;
        }
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
    void setLeft(TreeNode<InputC, OutputC>* l)
    {
        left = l;
    }

    /**
     * Set the right child
     * @param r a pointer to the right child node
     */
    void setRight(TreeNode<InputC, OutputC>* r)
    {
        right = r;
    }

    /**
     * This method is used to determine if the object is a leaf or an
     * internal node
     * @return true if it is a leaf, false otherwise
     */
    virtual bool isLeaf();

    /**
     * Empty destructor
     */
    virtual ~InternalTreeNode()
    {
        if (left != nullptr)
        {
            delete left;
        }
        if (right != nullptr)
        {
            delete right;
        }
    }

    /**
     *
     */
    virtual void writeOnStream(ofstream& out)
    {
        out << "N" << endl;
        out << axis << " " << split;
        out << endl;
        if (left)
        {
            out << *left;
        }
        else
        {
            out << "Empty" << endl;
        }
        if (right)
        {
            out << *right;
        }
        else
        {
            out << "Empty" << endl;
        }
    }

    /**
     *
     */
    virtual void readFromStream (ifstream& in)
    {
        //TODO implement
        /*string type;
        TreeNode<InputC>* children[2];
        in >> axis >> split;
        for (unsigned int i = 0; i < 2; i++)
        {
            in >> type;
            if ("L" == type)
            {
                children[i] = new rtLeaf();
            }
            else if ("LS" == type)
            {
                children[i] = new rtLeafSample();
            }
            else if ("LLI" == type)
            {
                children[i] = new rtLeafLinearInterp();
            }
            else if ("N" == type)
            {
                children[i] = new InternalTreeNode();
            }
            else
            {
                children[i] = 0;
            }

            if (children[i])
            {
                in >> *children[i];
            }
        }
        left = children[0];
        right = children[1];*/
    }

private:
    int axis;  // the axis of split
    double split;  // the value of split
    TreeNode<InputC, OutputC>* left;  // pointer to right child
    TreeNode<InputC, OutputC>* right;  // pointer to left child
};


}


#endif /* INCLUDE_RELE_APPROXIMATORS_REGRESSORS_TREES_INTERNALTREENODE_H_ */