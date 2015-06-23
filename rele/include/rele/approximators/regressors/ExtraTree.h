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
#include "RandomGenerator.h"

#include <stdexcept>


//FIXME LEVARE STO SCHIFO DI MACRO!!!!
// #define FEATURE_PROPAGATION

#define SPLIT_UNIFORM
#define SPLIT_VARIANCE
// #define VAR_RED_CORR

namespace ReLe
{

template<class InputC, class OutputC>
class ExtraTree: public RegressionTree<InputC, OutputC>
{
    using RegressionTree<InputC, OutputC>::root;

    enum AttributeState
    {
        Selectable,
        NotSelectable,
        Selected
    };

public:
    /**
     * Basic constructor
     * @param ex a vector containing the training set
     * @param k number of selectable attributes to be randomly picked
     * @param nmin minimum number of tuples in a leaf
     */
    ExtraTree(unsigned int input_size = 1, unsigned int output_size = 1, int k =
                  5, int nmin = 2, double score_th = 0.0)
    {
        root = NULL;
        mNumSplits = k;
        mNMin = nmin;
        mFeatureRelevance = nullptr;
        mScoreThreshold = score_th;
    }

    /**
     * Empty destructor
     */
    virtual ~ExtraTree()
    {

    }

    /**
     * Initialize data structures for feature ranking
     */
    void initFeatureRanks()
    {
        if (mFeatureRelevance == nullptr)
        {
            mFeatureRelevance = new double[phi.rows()];
        }
        for (unsigned int i = 0; i < phi.rows(); i++)
        {
            mFeatureRelevance[i] = 0.0;
        }
    }

    /**
     * Builds an approximation model for the training set
     * @param  data The training set
     * @param   overwrite When several training steps are run on the same inputs,
     it can be more efficient to reuse some structures.
     This can be done by setting this parameter to false
     */
    virtual void train(const BatchDataset_<InputC, OutpuC>& ds)
    {
        this->cleanTree();
        root = buildExtraTree(ds);
    }

    /**
     *
     */
    virtual void writeOnStream(std::ofstream& out)
    {
        //TODO implement
    }

    /**
     *
     */
    virtual void readFromStream(std::ifstream& in)
    {
        //TODO implement
    }

    /**
     *
     */
    double* evaluateFeatures()
    {
        return mFeatureRelevance;
    }

private:
    /**
     * This method build the Extra Tree
     * @param ex the vector containing the training set
     */
    TreeNode<InputC>* buildExtraTree(const BatchData<InputC, OutputC>& ds)
    {
        /*************** part 1 - END CONDITIONS ********************/
        int size = ds.size(); //size of dataset
        // END CONDITION 1: return a leaf if |ex| is less than nmin
        if (size < mNMin)
        {
            // 		cout << "size = " << size << endl;
            if (size == 0)
            {
                return 0;    //EMPTYLEAF
            }
            else
            {
                if (mLeafType == CONSTANT)
                {
                    return new LeafTreeNode<InputC, OutputC>(ds);
                }
                else
                {
                    //return new rtLeafLinearInterp(ds);
                    //TODO implement
                }
            }
        }
        // END CONDITION 2: return a leaf if all output variables are equals
        bool eq = true;
        const OutputC& checkOut = ds.getOutput(0);
        for (int i = 1; i < size && eq; i++)
        {
            const OutputC& newOut = ds.getOutput(i);
            if (output_traits<OutputC>::isAlmostEqual(checkOut, newOut))
            {
                eq = false;
                break;
            }
        }
        if (eq)
        {
            if (mLeafType == CONSTANT)
            {
                return new LeafTreeNode<InputC, OutputC>(ds);
            }
            else
            {
                //return new rtLeafLinearInterp(ds);
                //TODO implement
            }
        }

        unsigned int attnum = phi.rows(); //number of attributes
        std::vector<bool> constant(attnum, true); //indicates if and attribute is costant (true) or not (false)
        int end = attnum; //number of true values in constant
        arma::vec&& input = phi(ds.getInput(0));

        //check if the attributes are constant and build constant vector
        eq = true;
        for (int i = 1; i < size && end > 0; i++)
        {
            arma::vec&& otherInput = phi(ds.getInput(i));
            for (int c = 0; c < attnum; c++)
            {
                if (constant[c] && input[c] != otherInput[c])
                {
                    constant[c] = false;
                    end--;
                    eq = false;
                }
            }
        }

        // END CONDITION 3: return a leaf if all input variables are equals
        if (eq)
        {
            if (mLeafType == CONSTANT)
            {
                return new rtLeaf(ds);
            }
            else
            {
                return new rtLeafLinearInterp(ds);
            }
        }

        /************** part 2 - TREE GENERATIONS *******************/
        //now we have a vector (costant) that indicates if
        //an attribute is constant in every example;
        //selected will indicate if an attribute is selectable to be
        //splitted
        std::vector<AttributeState> attributesState(attnum);
        int selectable = 0;
        for (int c = 0; c < attnum; c++)
        {
            //it will avoid the selection of costant attributes
            if (constant[c])
            {
                attributesState[c] = NotSelectable;
            }
            else
            {
                attributesState[c] = Selectable;
                selectable++;
            }
        }

        //if the number of selectable attributes is <= k, all of
        //them will be candidate to split, else they will randomly
        //selected
        unsigned int candidates_size =
            selectable <= mNumSplits ? selectable : mNumSplits;
        std::vector<unsigned int> candidates(candidates_size);
        unsigned int num_candidates = candidates_size;

        while (num_candidates > 0)
        {
            unsigned int r = (rand() % num_candidates);
            unsigned int sel_attr = 0;
            while (attributesState[sel_attr] != Selectable || r > 0)
            {
                if (attributesState[sel_attr] == Selectable)
                {
                    r--;
                }

                sel_attr++;
            }

            candidates[num_candidates - 1] = sel_attr;
            attributesState[sel_attr] = Selected;
            num_candidates--;
        }

        //generate the first split
        int bestattribute = candidates[0]; //best attribute (indicated by number) found
        double bestsplit = pickRandomSplit(ds, candidates[0]); //best split value

        // split inputs in two subsets
        std::vector<unsigned int> indexesLow;
        std::vector<unsigned int> indexesHigh;
        this->splitDataset(ds, bestattribute, bestsplit, indexesLow, indexesHigh);

        MiniBatchData<InputC, OutputC> bestLow(ds,indexesLow);
        MiniBatchData<InputC, OutputC> bestHigh(ds,indexesHigh);

        double bestscore = score(ds, bestLow, bestHigh);

        //generates remaining splits and overwrites the actual best if better one is found
        for (unsigned int c = 1; c < candidates_size; c++)
        {
            double split = pickRandomSplit(ds, candidates[c]);

            indexesLow.clear();
            indexesHigh.clear();
            this->splitDataset(ds, bestattribute, bestsplit, indexesLow, indexesHigh);

            MiniBatchData<InputC, OutputC> low(ds,indexesLow);
            MiniBatchData<InputC, OutputC> high(ds,indexesHigh);

            double s = score(ds, low, high);
            //check if a better split was found
            if (s > bestscore)
            {
                bestscore = s;
                bestsplit = split;
                bestSl = low;
                bestSr = high;
                bestattribute = candidates[c];
            }
        }

        //    cout << "Best: " << bestattribute << " " << bestscore << " " << mScoreThreshold << endl;
        if (bestscore < mScoreThreshold)
        {
            if (mLeafType == CONSTANT)
            {
                return new LeafTreeNode<InputC, OutputC>(ds);
            }
            else
            {
                //return new rtLeafLinearInterp(ds);
                //TODO implement
            }
        }
        else
        {
            if (mFeatureRelevance != nullptr)
            {
                double variance_reduction = varianceReduction(ds, bestLow,
                                            bestHigh) /* ds.size() * ds->Variance()*/;
#ifdef FEATURE_PROPAGATION
                mSplittedAttributes.insert(bestattribute);
                mSplittedAttributesCount.insert(bestattribute);
                set<int>::iterator it;
                for (it = mSplittedAttributes.begin(); it != mSplittedAttributes.end(); ++it)
                {
                    mFeatureRelevance[*it] += variance_reduction / (double)mSplittedAttributes.size();
                }
#else
                mFeatureRelevance[bestattribute] += variance_reduction;
#endif
            }

            //build the left and the right children
            TreeNode<OutputC>* left = BuildExtraTree(bestLow);
            TreeNode<OutputC>* right = BuildExtraTree(bestHigh);

#ifdef FEATURE_PROPAGATION
            if (mFeatureRelevance != NULL)
            {
                mSplittedAttributesCount.erase(mSplittedAttributesCount.find(bestattribute));
                if (mSplittedAttributesCount.find(bestattribute) == mSplittedAttributesCount.end())
                {
                    mSplittedAttributes.erase(bestattribute);
                }
            }
#endif
            //return the current node
            return new InternalTreeNode<OutputC>(bestattribute, bestsplit, left, right);
        }
    }


    /**
     * This method picks a split randomly choosen such that it's greater than the minimum
     * observations of vector ex and it's less or equal than the maximum one
     * @param ex the vector containing the observations
     * @param attsplit number of attribute to split
     * @return the split value
     */
    double pickRandomSplit(const BatchData<InputC, outputC>& ds, int attsplit)
    {
#ifdef SPLIT_UNIFORM
        double min, max, tmp;

        //initialize min and max with the attribute value of the first observation
        min = phi(ds.getInput(0))[attsplit];
        max = min;

        //looking for min and max value of the dataset
        for (unsigned int c = 1; c < ds.size(); c++)
        {
            tmp = phi(ds.getInput(c))[attsplit];
            if (tmp < min)
            {
                min = tmp;
            }
            else if (tmp > max)
            {
                max = tmp;
            }
        }
        //return a value in (min, max]
        return sampleUniformHigh(min, max);
#else
        unsigned int r = rand() % ds->size();
        double value = ds->at(r)->GetInput(attsplit);
        double previous = value, next = value;
        for (unsigned int c = 0; c < ds->size(); c++)
        {
            double tmp = ds->at(c)->GetInput(attsplit);
            if (tmp < value && tmp > previous)
            {
                previous = tmp;
            }
            else if (tmp > value && tmp < next)
            {
                next = tmp;
            }
            else if (previous == value && tmp < value)
            {
                previous = tmp;
            }
            else if (next == value && tmp > value)
            {
                next = tmp;
            }
        }
//   cout << "R = " << r << " out of " << ds->size() << endl;
        return sampleUniformHigh(previous, next);
#endif
    }

    /**
     * This method computes the variance reduction on splitting a dataset
     * @param ds original dataset
     * @param dsl one partition
     * @param dsr the other one
     * @return the percentage of variance
     */
    double varianceReduction(const BatchData<InputC, OutputC>& ds,
    						 const BatchData<InputC, OutputC>& dsl,
							 const BatchData<InputC, OutputC>& dsr)
    {
        // VARIANCE REDUCTION
        double corr_fact_dsl = 1.0, corr_fact_dsr = 1.0, corr_fact_ds = 1.0;
#ifdef VAR_RED_CORR
        if (dsl->size() > 1)
        {
            corr_fact_dsl = (double)(dsl->size() / (dsl->size() - 1));
            corr_fact_dsl *= corr_fact_dsl;
        }
        if (dsr->size() > 1)
        {
            corr_fact_dsr = (double)(dsr->size() / (dsr->size() - 1));
            corr_fact_dsr *= corr_fact_dsr;
        }
        corr_fact_ds = (double)(ds->size() / (ds->size() - 1));
        corr_fact_ds *= corr_fact_ds;
#endif
        if (ds.size() == 0 /*|| ds->Variance() == 0.0*/)
        {
            return 0.0;
        }
        else
        {
        	//FIXME
            return 1 ;//- ((double)corr_fact_dsl * dsl->size() * dsl->Variance() + (double)corr_fact_dsr * dsr->size() * dsr->Variance()) / ((double)corr_fact_ds * ds->size() * ds->Variance());
        }
    }

    /**
     * This method computes the probability that two partition of a dataset has different means
     * @param ds original dataset
     * @param dsl one partition
     * @param dsr the other one
     * @return the probability value
     */
    double probabilityDifferentMeans(const BatchData<InputC, OutputC>& ds,
				 const BatchData<InputC, OutputC>& dsl,
				 const BatchData<InputC, OutputC>& dsr)
    {
        if (ds.size() == 0)
            return 0.0;

        double score = 0.0;
        //TODO implement
//        // T-STUDENT
//        double mean_diff = fabs(dsl->Mean() - dsr->Mean());
//        //   if (dsl->size() == 0 || dsr->size() == 0)
//        double size_dsl = (double)dsl->size() - 1.0;
//        double size_dsr = (double)dsr->size() - 1.0;
//        if (size_dsl < 1.0 && size_dsr < 1.0)
//        {
//            //     cout << "Score = 0.0 (one set empty)" << endl;
//            return 1.0;
//        }
//        else if (size_dsl < 1.0)
//        {
//            score = 2 * (gsl_cdf_tdist_P (mean_diff / sqrtf(dsr->Variance() / size_dsr), size_dsr) - 0.5);
//            if (score >= 1.0)
//            {
//                score += mean_diff / sqrtf(dsr->Variance() / size_dsr);
//            }
//            //     cout << "Score = " << score << " (empty set)" << endl;
//            return score;
//        }
//        else if (size_dsr < 1.0)
//        {
//            score = 2 * (gsl_cdf_tdist_P (mean_diff / sqrtf(dsl->Variance() / size_dsl), size_dsl) - 0.5);
//            if (score >= 1.0)
//            {
//                score += mean_diff / sqrtf(dsl->Variance() / size_dsl);
//            }
//            //     cout << "Score = " << score << " (empty set)" << endl;
//            return score;
//        }
//        double dsl_mean_variance = dsl->Variance() / size_dsl;
//        double dsr_mean_variance = dsr->Variance() / size_dsr;
//        double mean_diff_variance = dsl_mean_variance + dsr_mean_variance;
//        if (mean_diff_variance < 1e-6)
//        {
//            if (mean_diff > 1e-6)
//            {
//                //       cout << "Score = 1.0 (two constant sets)" << endl;
//                return 1.0;
//            }
//            else
//            {
//                //       cout << "Score = 0.0 (splitting a constant)" << endl;
//                return 0.0;
//            }
//        }
//
//        double dof = mean_diff_variance * mean_diff_variance
//                     / (  dsl_mean_variance * dsl_mean_variance / size_dsl
//                          + dsr_mean_variance * dsr_mean_variance / size_dsr);
//        score = gsl_cdf_tdist_P (mean_diff / sqrtf(mean_diff_variance), dof);
//        score = 2.0 * (score - 0.5);
//        if (score >= 1.0)
//        {
//            score += mean_diff / sqrtf(mean_diff_variance);
//        }
//        //   cout << " mean diff = " << mean_diff << " sqrt mean diff variance = " << sqrtf(mean_diff_variance) << " dof = " << dof << endl;
//        //   cout << "Score = " << score << endl;
        return score;
    }

    /**
     * This method compute the score (relative variance reduction) given by a split
     * @param s the vector containing the observations
     * @param sl the left partition of the observations set
     * @param sr the right partition of the observations set
     * @return the score
     */
    double score(Dataset* ds, Dataset* dsl, Dataset* dsr)
    {
#ifdef SPLIT_VARIANCE
        return varianceReduction(ds, dsl, dsr);
#else
        return probabilityDifferentMeans(ds, dsl, dsr);
#endif
    }

private:
    int mNMin; //minimum number of tuples for splitting
    int mNumSplits; //number of selectable attributes to be randomly picked
    double* mFeatureRelevance; //array of relevance values of input feature
    double mScoreThreshold;
    //multiset<int> mSplittedAttributesCount; FIXME
    set<int> mSplittedAttributes;
};

}

#endif /* INCLUDE_RELE_APPROXIMATORS_REGRESSORS_EXTRATREE_H_ */
