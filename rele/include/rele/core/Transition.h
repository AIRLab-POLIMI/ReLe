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

#ifndef INCLUDE_RELE_CORE_TRANSITION_H_
#define INCLUDE_RELE_CORE_TRANSITION_H_

#include <vector>
#include <fstream>

#include "BasisFunctions.h"

namespace ReLe
{
template<class ActionC, class StateC>
struct Transition
{
    static_assert(std::is_base_of<Action, ActionC>::value, "Not valid Action class as template parameter");
    static_assert(std::is_base_of<State, StateC>::value, "Not a valid State class as template parameter");

    StateC x;
    ActionC u;
    StateC xn;
    Reward r;

    void init(const StateC& x)
    {
        this->x = x;
    }

    void update(const ActionC& u, const StateC& xn, const Reward& r)
    {
        this->u = u;
        this->xn = xn;
        this->r = r;
    }


    void print(std::ostream& os)
    {
        os << "0,0,"
           << x  << ","
           << u  << ","
           << r  << std::endl;
    }

    void printLast(std::ostream& os)
    {
        os  << "1,"
            << xn.isAbsorbing() << ","
            << xn << std::endl;
    }
};

template <class ActionC, class StateC>
class Episode : public std::vector<Transition<ActionC,StateC>>
{

public:
    void print(std::ostream& os)
    {
        for(auto& sample : *this)
        {
            sample.print(os);
        }

        this->back().printLast(os);
    }

};

template<class ActionC, class StateC>
class Dataset : public std::vector<Episode<ActionC,StateC>>
{

public:
    arma::mat computefeatureExpectation(AbstractBasisMatrix& basis, double gamma = 1)
    {
        size_t episodes = this->size();
        arma::mat featureExpectation(basis.rows(), basis.cols(), arma::fill::zeros);

        for(auto& episode : *this)
        {
            arma::mat episodefeatureExpectation(basis.rows(), basis.cols(), arma::fill::zeros);;

            for(unsigned int t = 0; t < episode.size(); t++)
            {
                Transition<ActionC, StateC>& transition = episode[t];
                episodefeatureExpectation += std::pow(gamma, t) * basis(transition.x);
            }

            Transition<ActionC, StateC>& transition = episode.back();
            episodefeatureExpectation += std::pow(gamma, episode.size() + 1) * basis(transition.xn);


            featureExpectation += episodefeatureExpectation;
        }

        featureExpectation /= episodes;

        return featureExpectation;
    }

    void addData(Dataset<ActionC, StateC>& data)
    {
        this->insert(this->data.end(), data.begin(), data.end());
    }

    void setData(Dataset<ActionC, StateC>& data)
    {
        this->erase();
        addData(data);
    }


public:
    void writeToStream(std::ostream& os) //FIXME change this with new file format
    {
        int i, nbep = this->size();

        if (nbep > 0)
        {

            Transition<ActionC, StateC>& sample = (*this)[0][0];
            os << sample.x.serializedSize()  << ","
               << sample.u.serializedSize()  << ","
               << sample.r.size()  << std::endl;

            for (i = 0; i < nbep; ++i)
            {
                Episode<ActionC,StateC>& samples = this->at(i);
                samples.print(os);
            }
        }
    }


};


}

#endif /* INCLUDE_RELE_CORE_TRANSITION_H_ */
