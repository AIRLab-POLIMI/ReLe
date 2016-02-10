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
#include <armadillo>

namespace ReLe
{

template<class OutputC>
class TreeNode
{
public:

    TreeNode() {}

    virtual ~TreeNode() {}

    virtual bool isLeaf()
    {
        return false;
    }

    virtual bool isEmpty()
    {
        return false;
    }

    virtual int getAxis()
    {
        return -1;
    }

    virtual OutputC getValue(const arma::vec& input) = 0;

    virtual double getSplit()
    {
        return -1;
    }

    virtual TreeNode<OutputC>* getLeft()
    {
        return nullptr;
    }

    virtual TreeNode<OutputC>* getRight()
    {
        return nullptr;
    }

    virtual void writeOnStream (std::ofstream& out) = 0;

    virtual void readFromStream (std::ifstream& in) = 0;

    friend std::ofstream& operator<< (std::ofstream& out, TreeNode<OutputC>& n)
    {
        n.writeOnStream(out);
        return out;
    }

    friend std::ifstream& operator>> (std::ifstream& in, TreeNode<OutputC>& n)
    {
        n.readFromStream(in);
        return in;
    }

};


}

#endif /* INCLUDE_RELE_APPROXIMATORS_REGRESSORS_TREES_TREENODE_H_ */
