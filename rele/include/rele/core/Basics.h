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

#ifndef BASICS_H_
#define BASICS_H_

#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include <armadillo>

namespace ReLe
{

struct EnvirormentSettings
{
    bool isFiniteHorizon;
    double gamma;
    unsigned int horizon;

    bool isAverageReward;
    bool isEpisodic;

    size_t finiteStateDim;
    unsigned int finiteActionDim;

    unsigned int continuosStateDim;
    unsigned int continuosActionDim;

    unsigned int rewardDim;

    inline void WriteToStream(std::ostream& out) const
    {
        out << finiteStateDim << "\t" << finiteActionDim << std::endl;
        out << continuosStateDim << "\t" << continuosActionDim << std::endl;
        out << rewardDim << std::endl;
        out << gamma << "\t" << isFiniteHorizon << "\t" << horizon << "\t";
        out << isEpisodic << "\t" << isAverageReward << std::endl;
    }

    inline void ReadFromStream(std::istream& in)
    {
        in >> finiteStateDim >> finiteActionDim;
        in >> continuosStateDim >> continuosActionDim;
        in >> rewardDim;
        in >> gamma >> isFiniteHorizon >> horizon;
        in >> isEpisodic >> isAverageReward;
    }
};

class Action
{
public:
    inline virtual std::string to_str() const
    {
        return "action";
    }

    inline virtual ~Action()
    {

    }
};

class FiniteAction: public Action
{
public:
    FiniteAction()
    {
        actionN = 0;
    }

    inline unsigned int getActionN() const
    {
        return actionN;
    }

    inline void setActionN(unsigned int actionN)
    {
        this->actionN = actionN;
    }

    inline virtual std::string to_str() const
    {
        return "u = " + std::to_string(actionN);
    }

    inline virtual ~FiniteAction()
    {

    }

private:
    unsigned int actionN;
};

class DenseAction: public Action, public arma::vec
{
public:
    DenseAction ()
    { }

    DenseAction(std::size_t size) :
        arma::vec(size)
    {    }

    DenseAction(arma::vec& other) :
        arma::vec(other.n_elem)
    {
        this->set_size(other.n_elem);
        for (int i = 0; i < other.n_elem; ++i)
            this->at(i) = other[i];
    }

    inline virtual std::string to_str() const
    {
        const arma::vec& self = *this;
        std::stringstream ss;
        ss << "u = [";

        size_t i;
        for (i = 0; i + 1 < self.n_elem; i++)
            ss << self[i] << ", ";

        ss << self[i] << "]";

        return ss.str();
    }

    inline virtual ~DenseAction()
    {

    }

    inline virtual void copy_vec(const arma::vec& other)
    {
        this->set_size(other.n_elem);
        for (int i = 0; i < other.n_elem; ++i)
            this->at(i) = other[i];
    }

    inline virtual bool isAlmostEqual(const arma::vec& other, double epsilon = 1e-6) const
    {
        int i, dim = this->n_elem;
        if (other.n_elem != dim)
            return false;
        for (i = 0; i < dim; ++i)
            if (fabs(this->at(i) - other[i]) > epsilon)
                return false;

        return true;
    }
};

class State
{
public:
    State() :
        absorbing(false)
    {

    }

    inline bool isAbsorbing() const
    {
        return absorbing;
    }

    inline void setAbsorbing(bool absorbing = true)
    {
        this->absorbing = absorbing;
    }

    inline virtual std::string to_str() const
    {
        return "state";
    }

    inline virtual ~State()
    {

    }

private:
    bool absorbing;
};

class FiniteState: public State
{
public:
    FiniteState()
    {
        stateN = 0;
    }

    inline std::size_t getStateN() const
    {
        return stateN;
    }

    inline void setStateN(std::size_t stateN)
    {
        this->stateN = stateN;
    }

    inline virtual std::string to_str() const
    {
        return "x = " + std::to_string(stateN);
    }

    inline virtual ~FiniteState()
    {

    }

private:
    std::size_t stateN;

};

class DenseState: public State, public arma::vec
{
public:
    DenseState()
    {
    }

    DenseState(std::size_t size) :
        arma::vec(size)
    {
    }

    inline virtual std::string to_str() const
    {
        const arma::vec& self = *this;
        std::stringstream ss;
        ss << "x = [";

        size_t i;
        for (i = 0; i + 1 < self.n_elem; i++)
            ss << self[i] << ", ";

        ss << self[i] << "]";

        return ss.str();
    }

    inline virtual ~DenseState()
    {

    }

};

typedef std::vector<double> Reward;

inline std::ostream& operator<<(std::ostream& os, const Action& action)
{
    os << action.to_str();
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const State& state)
{
    os << state.to_str();
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const Reward& reward)
{
    os << "r = [";

    size_t i;
    for (i = 0; i + 1 < reward.size(); i++)
        os << reward[i] << ", ";

    os << reward[i] << "]";
    return os;
}

}

#endif /* BASICS_H_ */
