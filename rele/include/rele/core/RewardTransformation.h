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

#ifndef REWARDTRANSFORMATION_H_
#define REWARDTRANSFORMATION_H_

#include "rele/core/Basics.h"
#include <cassert>

namespace ReLe
{

/*!
 * Basic interface for reward function scalarization.
 */
class RewardTransformation
{
public:
    /*!
     * Evaluation method
     * \param r the reward vector
     * \return the scalarization of the reward vector
     */
    virtual double operator()(const Reward& r) = 0;
    virtual ~RewardTransformation() {}
};


/*!
 * This class implement the single objective scalarization, i.e. the
 * scalarization is done by selecting a single value of the reward function.
 */
class IndexRT : public RewardTransformation
{
public:
    /*!
     * Constructor.
     * \param idx the index of the reward element to choose
     */
    IndexRT(unsigned int idx)
        : index(idx)
    {
    }

    virtual inline double operator()(const Reward& r) override
    {
        return r[index];
    }

protected:
    unsigned int index;
};

/*!
 * This class implement the weighted sum scalarization, i.e. the
 * scalarization is a weighted sum of all elements in the reward
 * vector.
 */
class WeightedSumRT : public RewardTransformation
{
public:
    WeightedSumRT(arma::vec weights)
        : weights(weights)
    {
    }

    virtual inline double operator()(const Reward& r) override
    {
        double val = 0.0;
//        assert(r.size() == weights.n_elem);
        for (int i =0, ie = weights.n_elem; i < ie; ++i)
        {
            val += weights[i] * r[i];
        }
        return val;
    }

protected:
    arma::vec weights;
};

}//end namespace

#endif //REWARDTRANSFORMATION_H_
