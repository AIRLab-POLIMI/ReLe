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

#ifndef INCLUDE_RELE_APPROXIMATORS_REGRESSORS_REGRESSIONTREE_H_
#define INCLUDE_RELE_APPROXIMATORS_REGRESSORS_REGRESSIONTREE_H_

#include "Regressors.h"
#include "nodes/TreeNode.h"
#include "nodes/LeafTreeNode.h"
#include "nodes/InternalTreeNode.h"

namespace ReLe
{

template<class InputC, class OutputC>
class RegressionTree : public BatchRegressor_<InputC, OutputC>
{
public:
    virtual void train(const BatchData<InputC, OutputC>& dataset) = 0;

    /**
     * Get the root of the tree
     * @return a pointer to the root
     */
    TreeNode<InputC, OutputC>* getRoot()
    {
        return root;
    }

    virtual ~RegressionTree()
    {
        if(root != nullptr)
            delete root;
    }

protected:
    TreeNode<InputC, OutputC>* root;

};


}



#endif /* INCLUDE_RELE_APPROXIMATORS_REGRESSORS_REGRESSIONTREE_H_ */
