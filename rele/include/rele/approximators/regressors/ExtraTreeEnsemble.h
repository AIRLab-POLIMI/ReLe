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

#ifndef INCLUDE_RELE_APPROXIMATORS_REGRESSORS_EXTRATREEENSEMBLE_H_
#define INCLUDE_RELE_APPROXIMATORS_REGRESSORS_EXTRATREEENSEMBLE_H_

#include "ExtraTree.h"

namespace ReLe
{

template<class InputC, class OutputC>
class ExtraTreeEnsemble : public BatchRegressor_<InputC, OutputC>
{
public:
    /**
     * The basic constructor
     * @param m number of trees in the ensemble
     * @param k number of selectable attributes to be randomly picked
     * @param nMin minimum number of tuples in a leaf
     */
    ExtraTreeEnsemble(Features_<InputC>& phi, const EmptyTreeNode<OutputC>& emptyNode,
                      unsigned int outputSize = 1, unsigned int m = 50,
                      unsigned int k = 5, unsigned int nMin = 2,
                      double scoreThreshold = 0.0, LeafType leafType = Constant)
        : BatchRegressor_<InputC, OutputC>(phi, outputSize),
          phi(phi), emptyNode(emptyNode),
          m(m), k(k), nMin(nMin),
          scoreThreshold(scoreThreshold), leafType(leafType)
    {
        initialize();
    }

    /**
     *
     */
    virtual arma::vec operator() (const InputC& input) override
    {
        arma::vec output(this->outputDimension);
        output = evaluate(input);
        return output;
    }

    /**
     * Empty destructor
     */
    virtual ~ExtraTreeEnsemble()
    {
        cleanEnsemble();
    }

    /**
     * Initialize the ExtraTreeEnsemble by clearing the internal structures
     */
    virtual void initialize()
    {
        cleanEnsemble();
        ensemble.clear();
        for (unsigned int i = 0; i < m; i++)
        {
            auto tree = new ExtraTree<InputC, OutputC>(phi, emptyNode, leafType, this->outputDimension,
                    k, nMin, scoreThreshold);
            ensemble.push_back(tree);
        }
    }

    /**
     * Set nmin
     * @param nmin the minimum number of inputs for splitting
     */
    void setNMin(int nm)
    {
        nMin = nm;
    }

    /**
     * Builds an approximation model for the training set with parallel threads
     * @param  data The training set
     * @param   overwrite When several training steps are run on the same inputs,
                          it can be more efficient to reuse some structures.
                          This can be done by setting this parameter to false
     */
    virtual void train(const BatchData<InputC, OutputC>& dataset) override
    {
        for(auto tree : ensemble)
        {
            tree->train(dataset);
        }
    }

    void trainFeatures(const InputC& input, const OutputC& output) override
    {
        // TODO
    }

    /**
     * @return Value
     * @param  input The input data on which the model is evaluated
     */
    virtual OutputC evaluate(const InputC& input)
    {
        if (ensemble.size() == 0)
        {
            throw std::runtime_error("Empty ensemble evaluated");
        }

        OutputC out = ensemble[0]->evaluate(input);

        for(unsigned int i = 1; i < ensemble.size(); i++)
        {
            out += ensemble[i]->evaluate(input);
        }

        return out / static_cast<double>(ensemble.size());
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
    virtual void ReadFromStream(std::ifstream& in)
    {
        //TODO implement
    }

    /**
     * Initialize data structures for feature ranking
     */
    void initFeatureRanks()
    {
        for(auto tree : ensemble)
        {
            tree->initFeatureRanks();
        }
    }

private:
    void cleanEnsemble()
    {
        for (auto tree : ensemble)
        {
            if (tree)
            {
                delete tree;
            }
        }
    }

private:
    Features_<InputC>& phi;
    const EmptyTreeNode<OutputC>& emptyNode;
    unsigned int m; //number of trees in the ensemble
    unsigned int k; //number of selectable attributes to be randomly picked
    unsigned int nMin; //minimum number of tuples for splitting
    double scoreThreshold;
    LeafType leafType;

    std::vector<ExtraTree<InputC, OutputC>*> ensemble; //the extra-trees ensemble
};

}

#endif /* INCLUDE_RELE_APPROXIMATORS_REGRESSORS_EXTRATREEENSEMBLE_H_ */
