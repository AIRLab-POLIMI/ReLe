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

#ifndef INCLUDE_RELE_APPROXIMATORS_REGRESSORS_KDTREE_H_
#define INCLUDE_RELE_APPROXIMATORS_REGRESSORS_KDTREE_H_

#include "trees_bits/RegressionTree.h"

#include <stdexcept>

namespace ReLe
{

/**
 * This class implements kd-tree algorithm.
 * KD-Trees (K-Dimensional Trees) are a particular type of regression
 * trees, in fact this class extends the RegressionTree one.
 * In this method the regression tree is built from the training set by
 * choosing the cut-point at the local median of the cut-direction so
 * that the tree partitions the local training set into two subsets of
 * the same cardinality. The cut-directions alternate from one node to
 * the other: if the direction of cut is i j for the parent node, it is
 * equal to i j+1 for the two children nodes if j+1 < n with n the number
 * of possible cut-directions and i1 otherwise. A node is a leaf (i.e.,
 * is not partitioned) if the training sample corresponding to this node
 * contains less than nmin tuples. In this method the tree structure is
 * independent of the output values of the training sample.
 */
template<class InputC, class OutputC, bool denseOutput = true>
class KDTree: public RegressionTree<InputC, OutputC, denseOutput>
{
    USE_REGRESSION_TREE_MEMBERS

public:

    /**
     * Basic constructor
     * @param nm nmin, the minimum number of tuples for splitting
     */
    KDTree(Features_<InputC, denseOutput>& phi, const EmptyTreeNode<OutputC>& emptyNode,
           unsigned int output_size = 1, unsigned int nMin = 2)
        : RegressionTree<InputC, OutputC, denseOutput>(phi, emptyNode, output_size, nMin)
    {

    }

    /**
     * Empty destructor
     */
    virtual ~KDTree()
    {

    }

    virtual void trainFeatures(const BatchData_<OutputC, denseOutput>& featureDataset) override
    {
        this->cleanTree();
        root = buildKDTree(featureDataset, 0);
    }

    /**
     *
     */
    virtual void writeOnStream(std::ofstream& out)
    {
        out << *root;
    }

    /**
     *
     */
    virtual void readFromStream(std::ifstream& in)
    {
        //TODO [SERIALIZATION] implement
    }

private:

    /**
     * This method checks if all the inputs of a cut direction are constant
     * @param ex the vector containing the inputs
     * @param cutDir the cut direction
     * @return true if all the inputs are constant, false otherwise
     */
    double computeMedian(const BatchData_<OutputC, denseOutput>& ds, int cutDir)
    {
        std::vector<double> tmp;

        for (unsigned int i = 0; i < ds.size(); i++)
        {
            auto&& element = ds.getInput(i);
            tmp.push_back(element[cutDir]);
        }

        sort(tmp.begin(), tmp.end());
        tmp.erase(unique(tmp.begin(), tmp.end()), tmp.end());

        return tmp.at(tmp.size() / 2);

    }

    bool fixedInput(const BatchData_<OutputC, denseOutput>& ds, int cutDir)
    {
        if (ds.size() == 0)
        {
            return true;
        }

        auto&& element = ds.getInput(0);
        double val = element[cutDir];
        for (unsigned int i = 1; i < ds.size(); i++)
        {
            auto&& newElement = ds.getInput(i);
            double newVal = newElement[cutDir];
            if (std::abs(val - newVal) > THRESHOLD)
            {
                return false;
            }
        }

        return true;
    }

    /**
     * This method build the KD-Tree
     * @param ex the vector containing the training set
     * @param cutDir the current cut direction
     * @param store_sample allow to store samples into leaves
     * @return a pointer to the root
     */
    TreeNode<OutputC>* buildKDTree(const BatchData_<OutputC, denseOutput>& ds,
                                   int cutDir, bool store_sample = false)
    {
        unsigned int size = ds.size();
        /*****************part 1: end conditions*********************/
        if (size < nMin)
        {
            // if true -> leaf
            if (size == 0)
            {
                // if true -> empty leaf
                return &emptyNode;
            }
            else
            {
                return this->buildLeaf(ds, store_sample ? Samples : Constant);
            }
        }

        // control if inputs are all constants
        int cutTmp = cutDir;
        bool equal = false;
        while (fixedInput(ds, cutTmp) && !equal)
        {
            cutTmp = (cutTmp + 1) % ds.featuresSize();
            if (cutTmp == cutDir)
            {
                equal = true;
            }
        }

        // if constants create a leaf
        if (equal)
        {
            return this->buildLeaf(ds, store_sample ? Samples : Constant);
        }

        /****************part 2: generate the tree**************/
        //  begin operations to split the training set
        double cutPoint = computeMedian(ds, cutDir);

        arma::uvec indexesLow;
        arma::uvec indexesHigh;

        // split inputs in two subsets
        this->splitDataset(ds, cutDir, cutPoint, indexesLow, indexesHigh);

        BatchData_<OutputC, denseOutput>* lowEx = ds.cloneSubset(indexesLow);
        BatchData_<OutputC, denseOutput>* highEx = ds.cloneSubset(indexesHigh);

        // recall the method for left and right child
        TreeNode<OutputC>* left = buildKDTree(*lowEx, (cutDir + 1) % ds.featuresSize(), store_sample);
        TreeNode<OutputC>* right = buildKDTree(*highEx, (cutDir + 1) % ds.featuresSize(), store_sample);

        delete lowEx;
        delete highEx;

        // return the current node
        return new InternalTreeNode<OutputC>(cutDir, cutPoint, left, right);
    }

private:
    static constexpr double THRESHOLD = 1e-8;

};

}

#endif /* INCLUDE_RELE_APPROXIMATORS_REGRESSORS_KDTREE_H_ */
