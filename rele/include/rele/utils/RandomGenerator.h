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

#ifndef RANDOMGENERATOR_H_
#define RANDOMGENERATOR_H_

#include <random>

namespace ReLe
{

//-----------------------------------------------------------------------------
class RandomGenerator
{
public:
    inline static uint32_t randu32()
    {
        return gen();
    }

    inline static double sampleNormal()
    {
        std::normal_distribution<> dist;
        return dist(gen);
    }

    inline static double sampleNormal(double m, double sigma)
    {
        std::normal_distribution<> dist(m, sigma);
        return dist(gen);
    }

    /*inline static double sampleUniform()
     {
     std::uniform_01<> dist();
     return dist(gen);
     }*/

    inline static double sampleUniform(const double lo, const double hi)
    {
        std::uniform_real_distribution<> dist(lo, hi);
        return dist(gen);
    }

    inline static std::size_t sampleUniformInt(const int lo, const int hi)
    {
        std::uniform_int_distribution<> dist(lo, hi);
        return dist(gen);
    }

    inline static std::size_t sampleDiscrete(std::vector<double>& prob)
    {
        std::discrete_distribution<std::size_t> dist(prob.begin(),
                prob.end());
        return dist(gen);
    }

    template<class Iterator>
    inline static std::size_t sampleDiscrete(Iterator begin, Iterator end)
    {
        std::discrete_distribution<std::size_t> dist(begin, end);
        return dist(gen);
    }

    inline static bool sampleEvent(double prob)
    {
        std::uniform_real_distribution<> dist(0, 1);
        return dist(gen) < prob;
    }

private:
    //random generators
    static std::mt19937 gen;

};

}

#endif /* RANDOMGENERATOR_H_ */
