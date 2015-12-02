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
#include "nodes/EmptyTreeNode.h"
#include "Features.h"

namespace ReLe
{

template<class InputC, class OutputC>
class RegressionTree: public BatchRegressor_<InputC, OutputC>
{

public:
    RegressionTree(Features_<InputC>& phi,
                   const EmptyTreeNode<OutputC>& emptyNode,
                   unsigned int outputDimensions = 1,
                   unsigned int mNMin = 2) :
        BatchRegressor_<InputC, OutputC>(phi, outputDimensions),
        root(nullptr), emptyNode(emptyNode), nMin(mNMin), phi(phi) //FIXME regressors interface
    {

    }

    virtual arma::vec operator() (const InputC& input) override
    {
        arma::vec output(this->outputDimension);
        output = evaluate(input);
        return output;
    }

    /**
     * Evaluate the tree
     * @return OutputC
     * @param  input The input data on which the model is evaluated
     */
    virtual OutputC evaluate(const InputC& input)
    {
        if (!root)
        {
            throw std::runtime_error("Empty tree evaluated");
        }

        return root->getValue(phi(input));
    }

    /**
     * Set nMin
     * @param nm the minimum number of inputs for splitting
     */
    void setNMin(int nm)
    {
        nMin = nm;
    }

    /**
     * Get nmin
     */
    int getNMin()
    {
        return nMin;
    }

    virtual void train(const BatchData<InputC, OutputC>& dataset) override = 0;
    virtual void trainFeatures(typename input_collection<InputC>::const_ref_type input,
                               typename output_collection<OutputC>::const_ref_type output) override = 0;

    /**
     * Get the root of the tree
     * @return a pointer to the root
     */
    TreeNode<OutputC>* getRoot()
    {
        return root;
    }

    virtual ~RegressionTree()
    {
        cleanTree();
    }

protected:
    void cleanTree()
    {
        if (root && !root->isEmpty())
            delete root;
    }

    void splitDataset(const BatchData<InputC, OutputC>& ds,
                      int cutDir, double cutPoint,
                      std::vector<unsigned int>& indexesLow,
                      std::vector<unsigned int>& indexesHigh)
    {
        // split inputs in two subsets
        for (unsigned int i = 0; i < ds.size(); i++)
        {
            auto&& element = phi(ds.getInput(i));
            double tmp = element[cutDir];

            if (tmp < cutPoint)
            {
                indexesLow.push_back(i);
            }
            else
            {
                indexesHigh.push_back(i);
            }
        }
    }

    TreeNode<OutputC>* buildLeaf(const BatchData<InputC, OutputC>& ds, LeafType type)
    {
        switch(type)
        {
        case Constant:
            return new LeafTreeNode<InputC, OutputC>(ds);
        case Linear:
            return nullptr; //TODO implement
        case Samples:
            return new SampleLeafTreeNode<InputC, OutputC>(ds.clone());
        default:
            return nullptr;
        }
    }



protected:
    TreeNode<OutputC>* root;
    EmptyTreeNode<OutputC> emptyNode;
    Features_<InputC>& phi;

    unsigned int nMin;  // minimum number of tuples for splitting
};

#define USE_REGRESSION_TREE_MEMBERS               \
	typedef RegressionTree<InputC, OutputC> Base; \
	using Base::phi;                              \
    using Base::root;                             \
    using Base::emptyNode;                        \
    using Base::nMin;

}

#endif /* INCLUDE_RELE_APPROXIMATORS_REGRESSORS_REGRESSIONTREE_H_ */
