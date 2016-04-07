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

#include "rele/approximators/regressors/trees/trees_bits/RegressionTree.h"
#include "rele/utils/RandomGenerator.h"

#include <stdexcept>


//TODO [MACRO] remove, use subclass/strategy
// #define FEATURE_PROPAGATION

#define SPLIT_UNIFORM
#define SPLIT_VARIANCE
// #define VAR_RED_CORR

namespace ReLe
{

template<class InputC, class OutputC, bool denseOutput = true>
class ExtraTree: public RegressionTree<InputC, OutputC, denseOutput>
{
    USE_REGRESSION_TREE_MEMBERS

    enum AttributeState
    {
        Selectable,
        NotSelectable,
        Selected
    };

public:

    ExtraTree(Features_<InputC>& phi, const EmptyTreeNode<OutputC>& emptyNode, LeafType leafType = Constant,
              unsigned int output_size = 1, int k = 5, unsigned int nmin = 2, double score_th = 0.0)
        : RegressionTree<InputC, OutputC, denseOutput>(phi, emptyNode, output_size, nmin), leafType(leafType)
    {
        numSplits = k;
        scoreThreshold = score_th;
        computeFeaturerelevance = false;
    }

    virtual ~ExtraTree()
    {

    }

    void initFeatureRanks(unsigned int featureSize)
    {
        computeFeaturerelevance = true;
        featureRelevance.clear();
        featureRelevance.resize(featureSize, 0.0);
    }

    virtual void trainFeatures(const BatchData_<OutputC, denseOutput>& featureDataset) override
    {
        this->cleanTree();
        root = buildExtraTree(featureDataset);
    }

    virtual void writeOnStream(std::ofstream& out)
    {
        //TODO [SERIALIZATION] implement
    }

    virtual void readFromStream(std::ifstream& in)
    {
        //TODO [SERIALIZATION] implement
    }

    std::vector<double> evaluateFeatures()
    {
        return featureRelevance;
    }

private:

    TreeNode<InputC>* buildExtraTree(const BatchData_<OutputC, denseOutput>& ds)
    {
        /*************** part 1 - END CONDITIONS ********************/
        int size = ds.size(); //size of dataset
        // END CONDITION 1: return a leaf if |ex| is less than nMin
        if (size < nMin)
        {
            if (size == 0)
            {
                return &emptyNode;    //EMPTYLEAF
            }
            else
            {
                return this->buildLeaf(ds, leafType);
            }
        }
        // END CONDITION 2: return a leaf if all output variables are equals
        bool eq = true;
        const OutputC& checkOut = ds.getOutput(0);
        for (int i = 1; i < size && eq; i++)
        {
            const OutputC& newOut = ds.getOutput(i);
            if (!output_traits<OutputC>::isAlmostEqual(checkOut, newOut))
            {
                eq = false;
                break;
            }
        }
        if (eq)
        {
            return this->buildLeaf(ds, leafType);
        }

        unsigned int attnum = ds.featuresSize(); //number of attributes
        std::vector<bool> constant(attnum, true); //indicates if and attribute is costant (true) or not (false)
        int end = attnum; //number of true values in constant
        arma::vec&& input = ds.getInput(0);

        //check if the attributes are constant and build constant vector
        eq = true;
        for (int i = 1; i < size && end > 0; i++)
        {
            arma::vec&& otherInput = ds.getInput(i);
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
            return this->buildLeaf(ds, leafType);
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
            selectable <= numSplits ? selectable : numSplits;
        std::vector<unsigned int> candidates(candidates_size);
        unsigned int num_candidates = candidates_size;

        while (num_candidates > 0)
        {
            unsigned int r = RandomGenerator::sampleUniformInt(0, num_candidates-1);
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
        int bestAttribute = candidates[0]; //best attribute (indicated by number) found
        double bestSplit = pickRandomSplit(ds, candidates[0]); //best split value

        arma::uvec indexesLeftBest;
        arma::uvec indexesRightBest;

        // split inputs in two subsets
        this->splitDataset(ds, bestAttribute, bestSplit, indexesLeftBest, indexesRightBest);

        BatchData_<OutputC, denseOutput>* leftDs = ds.cloneSubset(indexesLeftBest);
        BatchData_<OutputC, denseOutput>* rightDs = ds.cloneSubset(indexesRightBest);

        double bestScore = score(ds, *leftDs, *rightDs);

        //generates remaining splits and overwrites the actual best if better one is found
        for (unsigned int c = 1; c < candidates_size; c++)
        {
            double split = pickRandomSplit(ds, candidates[c]);

            arma::uvec indexesLeft;
            arma::uvec indexesRight;
            this->splitDataset(ds, candidates[c], split, indexesLeft, indexesRight);

            double s = score(ds, *leftDs, *rightDs);

            //check if a better split was found
            if (s > bestScore)
            {
                bestScore = s;
                bestSplit = split;
                indexesLeftBest = indexesLeft;
                indexesRightBest = indexesRight;
                bestAttribute = candidates[c];
            }
        }

        //get the best split of two datasets

        //    cout << "Best: " << bestattribute << " " << bestscore << " " << mScoreThreshold << endl;
        if (bestScore < scoreThreshold)
        {
            delete leftDs;
            delete rightDs;

            return this->buildLeaf(ds, leafType);
        }
        else
        {
            if (computeFeaturerelevance)
            {
                double variance_reduction = varianceReduction(ds, *leftDs, *rightDs);
                variance_reduction *= ds.size() * arma::det(ds.getVariance());
#ifdef FEATURE_PROPAGATION
                mSplittedAttributes.insert(bestAttribute);
                mSplittedAttributesCount.insert(bestAttribute);
                set<int>::iterator it;
                for (it = mSplittedAttributes.begin(); it != mSplittedAttributes.end(); ++it)
                {
                    featureRelevance[*it] += variance_reduction / (double)mSplittedAttributes.size();
                }
#else
                featureRelevance[bestAttribute] += variance_reduction;
#endif
            }

            //build the left and the right children
            TreeNode<OutputC>* left = buildExtraTree(*leftDs);
            TreeNode<OutputC>* right = buildExtraTree(*rightDs);

#ifdef FEATURE_PROPAGATION
            if (featureRelevance != NULL)
            {
                mSplittedAttributesCount.erase(mSplittedAttributesCount.find(bestAttribute));
                if (mSplittedAttributesCount.find(bestAttribute) == mSplittedAttributesCount.end())
                {
                    mSplittedAttributes.erase(bestAttribute);
                }
            }
#endif

            delete leftDs;
            delete rightDs;

            //return the current node
            return new InternalTreeNode<OutputC>(bestAttribute, bestSplit, left, right);
        }
    }

    double pickRandomSplit(const BatchData_<OutputC, denseOutput>& ds, int attsplit)
    {
#ifdef SPLIT_UNIFORM
        double min, max, tmp;

        //initialize min and max with the attribute value of the first observation
        min = ds.getInput(0)[attsplit];
        max = min;

        //looking for min and max value of the dataset
        for (unsigned int c = 1; c < ds.size(); c++)
        {
            tmp = ds.getInput(c)[attsplit];
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
        return RandomGenerator::sampleUniformHigh(min, max);
#else
        unsigned int r = RandomGenerator::sampleUniform(0, ds.size());
        double value = ds.getInput(r)[attsplit];
        double previous = value, next = value;
        for (unsigned int c = 0; c < ds.size(); c++)
        {
            double tmp = ds.getInput(c)[attsplit];
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

        //return a value in (previous, next]
        return RandomGenerator::sampleUniformHigh(previous, next);
#endif
    }

    double varianceReduction(const BatchData_<OutputC, denseOutput>& ds,
                             const BatchData_<OutputC, denseOutput>& dsl,
                             const BatchData_<OutputC, denseOutput>& dsr)
    {
        // VARIANCE REDUCTION
        double corr_fact_dsl = 1.0, corr_fact_dsr = 1.0, corr_fact_ds = 1.0;
#ifdef VAR_RED_CORR
        if (dsl.size() > 1)
        {
            corr_fact_dsl = static_cast<double>(dsl.size() / (dsl.size() - 1));
            corr_fact_dsl *= corr_fact_dsl;
        }
        if (dsr.size() > 1)
        {
            corr_fact_dsr = static_cast<double>(dsr.size() / (dsr.size() - 1));
            corr_fact_dsr *= corr_fact_dsr;
        }
        corr_fact_ds = static_cast<double>(ds.size() / (ds.size() - 1));
        corr_fact_ds *= corr_fact_ds;
#endif
        if (ds.size() == 0 || arma::det(ds.getVariance()) < 1e-8)
        {
            return 0.0;
        }
        else
        {
            arma::mat varDS = corr_fact_ds * ds.size() * ds.getVariance();
            arma::mat varDSL = corr_fact_dsl * dsl.size() * dsl.getVariance();
            arma::mat varDSR = corr_fact_dsr * dsr.size() * dsr.getVariance();
            arma::mat I = arma::eye(varDS.n_rows, varDS.n_cols);
            return arma::det(I - (varDSL + varDSR) * arma::inv(varDS));
        }
    }

    double probabilityDifferentMeans(const BatchData_<OutputC, denseOutput>& ds,
                                     const BatchData_<OutputC, denseOutput>& dsl,
                                     const BatchData_<OutputC, denseOutput>& dsr)
    {
        if (ds.size() == 0)
            return 0.0;

        double score = 0.0;
        //TODO [MINOR] implement
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

    double score(const BatchData_<OutputC, denseOutput>& ds,
                 const BatchData_<OutputC, denseOutput>& dsl,
                 const BatchData_<OutputC, denseOutput>& dsr)
    {
#ifdef SPLIT_VARIANCE
        return varianceReduction(ds, dsl, dsr);
#else
        return probabilityDifferentMeans(ds, dsl, dsr);
#endif
    }

private:
    LeafType leafType;
    int numSplits; //number of selectable attributes to be randomly picked
    std::vector<double> featureRelevance; //array of relevance values of input feature
    bool computeFeaturerelevance;
    double scoreThreshold;
    //TODO [MINOR] implement features selection
    //multiset<int> mSplittedAttributesCount;
    //set<int> mSplittedAttributes;
};

}

#endif /* INCLUDE_RELE_APPROXIMATORS_REGRESSORS_EXTRATREE_H_ */
