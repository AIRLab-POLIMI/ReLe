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

#include "rele/approximators/regressors/trees/trees_bits/nodes/EmptyTreeNode.h"
#include "rele/approximators/regressors/trees/trees_bits/nodes/InternalTreeNode.h"
#include "rele/approximators/regressors/trees/trees_bits/nodes/LeafTreeNode.h"
#include "rele/approximators/regressors/trees/trees_bits/nodes/TreeNode.h"
#include "rele/approximators/Regressors.h"
#include "rele/approximators/Features.h"

namespace ReLe
{

template<class OutputC, bool denseInput>
class RegressionTree: public BatchRegressor_<OutputC, denseInput>
{
    using BatchRegressor_<OutputC, denseInput>::phi;
public:
    RegressionTree(unsigned int inputs,
                   const EmptyTreeNode<OutputC>& emptyNode,
                   unsigned int outputDimensions = 1,
                   unsigned int nMin = 2) :
        BatchRegressor_<OutputC, denseInput>(inputs, outputDimensions),
        root(nullptr), emptyNode(emptyNode), nMin(nMin)
    {

    }

    virtual OutputC operator() (const FeaturesType& input) override
    {
        if (!root)
            return emptyNode.getValue(input);

        return root->getValue(input);
    }

    virtual double computeJ(const BatchData_<OutputC, denseInput>& dataset) override
    {
        double J = 0;

        for(unsigned int i = 0; i < dataset.size(); i++)
        {
            OutputC yhat = root->getValue(dataset.getInput(i));
            OutputC y = dataset.getOutput(i);
            J += output_traits<OutputC>::errorSquared(yhat, y);
        }

        return J / dataset.size();
    }


    void setNMin(int nm)
    {
        nMin = nm;
    }


    int getNMin()
    {
        return nMin;
    }

    virtual void train(const BatchData_<OutputC, denseInput>& featureDataset) override = 0;


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

    void splitDataset(const BatchData_<OutputC, denseInput>& ds,
                      int cutDir, double cutPoint,
                      arma::uvec& indexesLow,
                      arma::uvec& indexesHigh)
    {
        indexesLow.set_size(ds.size());
        indexesHigh.set_size(ds.size());
        unsigned int lowNumber = 0;
        unsigned int highNumber = 0;

        // split inputs in two subsets
        for (unsigned int i = 0; i < ds.size(); i++)
        {
            auto&& element = ds.getInput(i);
            double tmp = element[cutDir];

            if (tmp < cutPoint)
            {
                indexesLow(lowNumber) = i;
                lowNumber++;
            }
            else
            {
                indexesHigh(highNumber) = i;
                highNumber++;
            }
        }

        indexesLow.resize(lowNumber);
        indexesHigh.resize(highNumber);
    }

    TreeNode<OutputC>* buildLeaf(const BatchData_<OutputC, denseInput>& ds, LeafType type)
    {
        switch(type)
        {
        case Constant:
            return new LeafTreeNode<OutputC, denseInput>(ds);
        case Linear:
            return nullptr; //TODO [MINOR] implement
        case Samples:
            return new SampleLeafTreeNode<OutputC, denseInput>(ds.clone());
        default:
            return nullptr;
        }
    }



protected:
    TreeNode<OutputC>* root;
    EmptyTreeNode<OutputC> emptyNode;

    unsigned int nMin;  // minimum number of tuples for splitting
};

#define USE_REGRESSION_TREE_MEMBERS               \
	typedef RegressionTree<OutputC, denseInput> Base; \
    using Base::root;                             \
    using Base::emptyNode;                        \
    using Base::nMin;

}

#endif /* INCLUDE_RELE_APPROXIMATORS_REGRESSORS_REGRESSIONTREE_H_ */
