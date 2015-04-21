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

#ifndef INCLUDE_RELE_POLICY_NONPARAMETRIC_TABULARPOLICY_H_
#define INCLUDE_RELE_POLICY_NONPARAMETRIC_TABULARPOLICY_H_

#include "Policy.h"

namespace ReLe
{

class TabularPolicy: public NonParametricPolicy<FiniteAction, FiniteState>
{

public:
    class updater;

public:
    virtual unsigned int operator()(const size_t& state);
    virtual double operator()(const size_t& state, const unsigned int& action);

    inline virtual std::string getPolicyName()
    {
        return "Tabular";
    }

    virtual std::string getPolicyHyperparameters()
    {
        return "";
    }

    virtual std::string printPolicy();

    updater update(size_t state)
    {
        return updater(pi.row(state));
    }

    inline void init(size_t nstates, unsigned int nactions)
    {
        pi.set_size(nstates, nactions);
        pi.fill(1.0/nactions);
    }

    virtual TabularPolicy* clone()
    {
        return new TabularPolicy(*this);
    }


public:
    class updater
    {
        updater(arma::subview_row<double>&& row);

    public:
        friend updater TabularPolicy::update(size_t state);

        void operator<<(double weight);

        inline bool toFill()
        {
            return currentIndex != nactions;
        }

        inline unsigned int getCurrentState()
        {
            return currentIndex;
        }

        void normalize();

    private:
        arma::subview_row<double> row;
        unsigned int nactions;
        unsigned int currentIndex;
    };

private:
    arma::mat pi;
};

}

#endif /* INCLUDE_RELE_POLICY_NONPARAMETRIC_TABULARPOLICY_H_ */
