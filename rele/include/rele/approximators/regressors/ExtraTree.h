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

#ifndef INCLUDE_RELE_APPROXIMATORS_REGRESSORS_EXTRATREE_H_
#define INCLUDE_RELE_APPROXIMATORS_REGRESSORS_EXTRATREE_H_

#include "trees/RegressionTree.h"

#include <stdexcept>

template<class InputC, class OutputC>
class ExtraTree
{
public:
    /**
     * Basic constructor
     * @param ex a vector containing the training set
     * @param k number of selectable attributes to be randomly picked
     * @param nmin minimum number of tuples in a leaf
     */
    ExtraTree(unsigned int input_size = 1,
              unsigned int output_size = 1,
              int k = 5,
              int nmin = 2,
              float score_th = 0.0);

    /**
     * Empty destructor
     */
    virtual ~ExtraTree();

    /**
     * Initialize data structures for feature ranking
     */
    void initFeatureRanks();

    /**
     * Set nmin
     * @param nmin the minimum number of inputs for splitting
     */
    void setNMin(int nm);

    /**
     * Builds an approximation model for the training set
     * @param  data The training set
     * @param   overwrite When several training steps are run on the same inputs,
                          it can be more efficient to reuse some structures.
                          This can be done by setting this parameter to false
     */
    virtual void train(const BatchDataset_<InputC, OutpuC>& ds, bool overwrite = true, bool normalize = true);

    /**
     * @return Value
     * @param  input The input data on which the model is evaluated
     */
    virtual OutputC evaluate(const arma::vec& input);

    /**
     *
     */
    virtual void writeOnStream (std::ofstream& out);

    /**
     *
     */
    virtual void readFromStream (std::ifstream& in);

    /**
     *
     */
    float* evaluateFeatures();

    /**
     *
     */
    void analyzeSplitCriteria(Dataset* ds);

private:
    /**
     * This method build the Extra Tree
     * @param ex the vector containing the training set
     */
    rtANode* BuildExtraTree (Dataset* ds);

    /**
     * This method traverses the tree to search the output belonging to a particular input
     * @param r the node to evaluate
     * @param in the input
     * @return the output value
     */
    float TraverseTree(rtANode* r, Tuple& in);

    /**
     * This method picks a split randomly choosen such that it's greater than the minimum
     * observations of vector ex and it's less or equal than the maximum one
     * @param ex the vector containing the observations
     * @param attsplit number of attribute to split
     * @return the split value
     */
    float PickRandomSplit(Dataset* ds, int attsplit);

    /**
     * This method partionates the input dataset in two datasets given an attribute att and a split value v:
     *  the right partition with elements that have a not greater than v and the left one with observations * with a less than v
     * @param ex the vector containing the input observations
     * @param left the partition with elements that have the attribute value less than the split value
     * @param right the partition with elements that have the attribute value greater (or equal) than the split value
     * @param attribute number of attribute to split (indicates a)
     * @param split the split value (indicates v)
     */
    void Partitionate(Dataset* ds, Dataset* left, Dataset* right, int attribute, float split);

    /**
     * This method computes the variance reduction on splitting a dataset
     * @param ds original dataset
     * @param dsl one partition
     * @param dsr the other one
     * @return the percentage of variance
     */
    float VarianceReduction(Dataset* ds, Dataset* dsl, Dataset* dsr);

    /**
     * This method computes the probability that two partition of a dataset has different means
     * @param ds original dataset
     * @param dsl one partition
     * @param dsr the other one
     * @return the probability value
     */
    float ProbabilityDifferentMeans(Dataset* ds, Dataset* dsl, Dataset* dsr);

    /**
     * This method compute the score (relative variance reduction) given by a split
     * @param s the vector containing the observations
     * @param sl the left partition of the observations set
     * @param sr the right partition of the observations set
     * @return the score
     */
    float Score(Dataset* ds, Dataset* dsl, Dataset* dsr);

private:
    int mNMin; //minimum number of tuples for splitting
    int mNumSplits; //number of selectable attributes to be randomly picked
    float* mFeatureRelevance; //array of relevance values of input feature
    float mScoreThreshold;
    LeafType mLeafType;
    multiset<int> mSplittedAttributesCount;
    set<int> mSplittedAttributes;
};



#endif /* INCLUDE_RELE_APPROXIMATORS_REGRESSORS_EXTRATREE_H_ */
