/*
 * rele,
 *
 *
 * Copyright (C) 2016 Davide Tateo
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

#ifndef INCLUDE_RELE_IRL_ALGORITHMS_EXECTEDDELTAIRL_H_
#define INCLUDE_RELE_IRL_ALGORITHMS_EXECTEDDELTAIRL_H_


namespace ReLe
{

template<class ActionC, class StateC>
class ExpectedDeltaIRL: public IRLAlgorithm<ActionC, StateC>
{
public:
    ExpectedDeltaIRL(Dataset<ActionC, StateC>& data,
                     DifferentiablePolicy<ActionC, StateC>& policy,
                     LinearApproximator& rewardf) : data(data), policy(policy), rewardf(rewardf)
    {

    }

    virtual ~ExpectedDeltaIRL()
    {

    }


    virtual void run() override
    {

    }

private:
    Dataset<ActionC, StateC>& data;
    DifferentiablePolicy<ActionC, StateC>& policy;
    LinearApproximator& rewardf;
};


}

#endif /* INCLUDE_RELE_IRL_ALGORITHMS_EXECTEDDELTAIRL_H_ */
