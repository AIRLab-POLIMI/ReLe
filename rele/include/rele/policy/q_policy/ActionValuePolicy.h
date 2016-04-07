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

#ifndef INCLUDE_RELE_POLICY_Q_POLICY_ACTIONVALUEPOLICY_H_
#define INCLUDE_RELE_POLICY_Q_POLICY_ACTIONVALUEPOLICY_H_

#include "rele/policy/Policy.h"
#include "rele/approximators/Regressors.h"

#include <type_traits>
#include <sstream>

namespace ReLe
{

template<class StateC>
struct q_type
{
    typedef std::add_pointer<void>::type type;
};

template<>
struct q_type<FiniteState>
{
    typedef std::add_pointer<arma::mat>::type type;
};

template<>
struct q_type<DenseState>
{
    typedef std::add_pointer<Regressor>::type type;
};

template<class StateC>
class ActionValuePolicy: public NonParametricPolicy<FiniteAction, StateC>
{
public:
    inline void setQ(typename q_type<StateC>::type Q)
    {
        this->Q = Q;
    }

    inline typename q_type<StateC>::type getQ()
    {
        return this->Q;
    }

    inline void setNactions(unsigned int nactions)
    {
        this->nactions = nactions;
    }

    virtual std::string printPolicy() override
    {
        return printPolicyWorker(static_cast<StateC*>(nullptr));
    }

    virtual ~ActionValuePolicy()
    {

    }

private:
    //TODO [CLEANUP] forse non è il meglio che si può fare, ma non voglio scrivere migliaia di classi. forse con un tratto?
    std::string printPolicyWorker(FiniteState*)
    {
        //TODO [MINOR] decidere come formattare l'output...
        std::stringstream ss;
        ss << "- Policy" << std::endl;
        for (unsigned int i = 0; i < Q->n_rows; i++)
        {
            arma::uword policy;
            Q->row(i).max(policy);
            ss << "policy(" << i << ") = " << policy << std::endl;
        }

        return ss.str();
    }

    template<class T>
    std::string printPolicyWorker(T*)
    {
        return "";
    }

protected:
    typename q_type<StateC>::type Q;
    unsigned int nactions;
};

}

#endif /* INCLUDE_RELE_POLICY_Q_POLICY_ACTIONVALUEPOLICY_H_ */
