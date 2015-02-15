/*
 * rele,
 *
 *
 * Copyright (C) 2015 Davide Tateo
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

#ifndef SAMPLEMANAGER_H_
#define SAMPLEMANAGER_H_

#include "BasicsTraits.h"
#include "Sample.h"

#include <map>
#include <vector>
#include <cassert>

namespace ReLe
{

template<class ActionC, class StateC>
class SampleManager
{
    static_assert(std::is_base_of<Action, ActionC>::value, "Not valid Action class as template parameter");
    static_assert(std::is_base_of<State, StateC>::value, "Not a valid State class as template parameter");

    typedef std::map<typename state_type<StateC>::type,
            std::map<typename action_type<ActionC>::type, double>> DeltaMap;
    typedef std::map<typename state_type<StateC>::type,
            std::map<typename action_type<ActionC>::type, arma::vec>> LambdaMap;
    typedef std::map<typename state_type<StateC>::type,
            std::map<typename action_type<ActionC>::type, int>> CountMap;

    typedef std::vector<Sample<ActionC, StateC>> SampleVector;

public:
    class DeltaFunctor
    {
    public:
        DeltaFunctor(DeltaMap& ndelta, CountMap& d) :
            ndelta(ndelta), d(d)
        {

        }

        inline double operator()(typename state_type<StateC>::type x,
                                 typename action_type<ActionC>::type u)
        {
            return ndelta[x][u] / d[x][u];
        }

    private:
        DeltaMap& ndelta;
        CountMap& d;
    };

public:
    SampleManager()
    {

    }

    DeltaFunctor getDelta(const arma::vec& theta)
    {
        for (auto& sample : samples)
        {
            if(ndelta.count(sample.x) == 0 || ndelta[sample.x].count(sample.u) == 0)
            {
                ndelta[sample.x][sample.u] = 0;
            }

            ndelta[sample.x][sample.u] += sample.r + theta[sample.xn]
                                          - theta[sample.x];
        }

        return DeltaFunctor(ndelta, d);

    }

    arma::vec lambda(typename state_type<StateC>::type x,
                     typename action_type<ActionC>::type u)
    {
        assert(nlambda.count(x) != 0 && nlambda[x].count(u) != 0);
        return nlambda[x][u] / d[x][u];
    }

    void addSample(Sample<ActionC, StateC>& sample)
    {
        samples.push_back(sample);

        if (d.count(sample.x) == 0 || d[sample.x].count(sample.u) == 0)
        {
            d[sample.x][sample.u] = 1;
            //FIXME implement features
            arma::vec phi(5);
            phi(sample.x) = 1;
            arma::vec phin(5);
            phin(sample.xn) = 1;
            nlambda[sample.x][sample.u] += phin - phi;
        }
        else
        {
            d[sample.x][sample.u]++;
            //FIXME implement features
            arma::vec phi(5);
            phi(sample.x) = 1;
            arma::vec phin(5);
            phin(sample.xn) = 1;
            nlambda[sample.x][sample.u] += phin - phi;
        }
    }

    void reset()
    {
        samples.clear();
        d.clear();
        ndelta.clear();
        nlambda.clear();
    }


    typename SampleVector::iterator begin()
    {
        return samples.begin();
    }

    typename SampleVector::iterator end()
    {
        return samples.end();
    }

private:
    DeltaMap ndelta;
    LambdaMap nlambda;
    CountMap d;
    SampleVector samples;

};

}


#endif /* SAMPLEMANAGER_H_ */
