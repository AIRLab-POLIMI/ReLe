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
#include "rele/approximators/regressors/Ensemble.h"


namespace ReLe
{

template<class InputC, class OutputC>
class ExtraTreeEnsemble_: public Ensemble_<InputC, OutputC>
{
public:
    ExtraTreeEnsemble_(Features_<InputC>& phi, const EmptyTreeNode<OutputC>& emptyNode,
                       unsigned int outputSize = 1, unsigned int nRegressors = 50,
                       unsigned int k = 5, unsigned int nMin = 2,
                       double scoreThreshold = 0.0, LeafType leafType = Constant)
        : Ensemble_<InputC, OutputC>(phi, outputSize), emptyNode(emptyNode),
          k(k), nMin(nMin), scoreThreshold(scoreThreshold),
          leafType(leafType)
    {
        initialize(nRegressors);
    }

    void initialize(unsigned int nRegressors)
    {
        this->cleanEnsemble();
        this->regressors.clear();
        for (unsigned int i = 0; i < nRegressors; i++)
        {
            auto tree = new ExtraTree<InputC, OutputC>(this->phi, emptyNode, leafType, this->outputDimension,
                    k, nMin, scoreThreshold);
            this->regressors.push_back(tree);
        }
    }

    void initFeatureRanks()
    {
        for(auto tree : this->regressors)
            tree->initFeatureRanks();
    }

    virtual void writeOnStream(std::ofstream& out) override
    {
        //TODO [SERIALIZATION] implement
    }

    virtual void readFromStream(std::ifstream& in) override
    {
        //TODO [SERIALIZATION] implement
    }

    virtual ~ExtraTreeEnsemble_()
    {

    }

private:
    const EmptyTreeNode<OutputC>& emptyNode;
    unsigned int k; // Number of selectable attributes to be randomly picked
    unsigned int nMin; // Minimum number of tuples for splitting
    double scoreThreshold;
    LeafType leafType;
};

typedef ExtraTreeEnsemble_<arma::vec, arma::vec> ExtraTreeEnsemble;

}

#endif /* INCLUDE_RELE_APPROXIMATORS_REGRESSORS_EXTRATREEENSEMBLE_H_ */
