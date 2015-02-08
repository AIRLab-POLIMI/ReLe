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

#include <boost/random.hpp>

namespace ReLe
{

//-----------------------------------------------------------------------------
// Xorshift RNG based on code by George Marsaglia
// http://en.wikipedia.org/wiki/Xorshift

class Xorshift
{
private:
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t w;

public:
    Xorshift()
    {
        reseed(uint32_t(0));
    }

    Xorshift(uint32_t seed)
    {
        reseed(seed);
    }

    void reseed(uint32_t seed)
    {
        x = 0x498b3bc5 ^ seed;
        y = 0;
        z = 0;
        w = 0;

        for (int i = 0; i < 10; i++)
            mix();
    }

    void reseed(uint64_t seed)
    {
        x = 0x498b3bc5 ^ (uint32_t) (seed >> 0);
        y = 0x5a05089a ^ (uint32_t) (seed >> 32);
        z = 0;
        w = 0;

        for (int i = 0; i < 10; i++)
            mix();
    }

    //-----------------------------------------------------------------------------

    void mix(void)
    {
        uint32_t t = x ^ (x << 11);
        x = y;
        y = z;
        z = w;
        w = w ^ (w >> 19) ^ t ^ (t >> 8);
    }

    uint32_t rand_u32(void)
    {
        mix();

        return x;
    }

    uint64_t rand_u64(void)
    {
        mix();

        uint64_t a = x;
        uint64_t b = y;

        return (a << 32) | b;
    }

    void rand_p(void * blob, int bytes)
    {
        uint32_t * blocks = reinterpret_cast<uint32_t*>(blob);

        while (bytes >= 4)
        {
            blocks[0] = rand_u32();
            blocks++;
            bytes -= 4;
        }

        uint8_t * tail = reinterpret_cast<uint8_t*>(blocks);

        for (int i = 0; i < bytes; i++)
        {
            tail[i] = (uint8_t) rand_u32();
        }
    }
};

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
        boost::random::normal_distribution<> dist;
        return dist(gen);
    }

    inline static double sampleNormal(double m, double sigma)
    {
        boost::random::normal_distribution<> dist(m, sigma);
        return dist(gen);
    }

    /*inline static double sampleUniform()
     {
     boost::random::uniform_01<> dist();
     return dist(gen);
     }*/

    inline static double sampleUniform(const double lo, const double hi)
    {
        boost::random::uniform_real_distribution<> dist(lo, hi);
        return dist(gen);
    }

    inline static std::size_t sampleUniformInt(const int lo, const int hi)
    {
        boost::random::uniform_int_distribution<> dist(lo, hi);
        return dist(gen);
    }

    inline static std::size_t sampleDiscrete(std::vector<double>& prob)
    {
        boost::random::discrete_distribution<std::size_t> dist(prob.begin(),
                prob.end());
        return dist(gen);
    }

    template<class Iterator>
    inline static std::size_t sampleDiscrete(Iterator begin, Iterator end)
    {
        boost::random::discrete_distribution<std::size_t> dist(begin, end);
        return dist(gen);
    }

    inline static bool sampleEvent(double prob)
    {
        boost::random::uniform_real_distribution<> dist(0, 1);
        return dist(gen) < prob;
    }

private:
    //random generators
    static boost::random::mt19937 gen;

};

}

#endif /* RANDOMGENERATOR_H_ */
