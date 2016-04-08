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
#include <iomanip>

#define OS_PRECISION 10

namespace ReLe
{

/*!
 * Basic struct to describe an environment
 */
struct EnvironmentSettings
{
    //! Finite horizon flag
    bool isFiniteHorizon;
    //! Average reward flag
    bool isAverageReward;
    //! Episodic environment flag
    bool isEpisodic;

    //! Discount factor
    double gamma;
    //! horizon of mdp
    unsigned int horizon;

    //! number of finite states of the mdp. Should be zero for continuos state spaces
    size_t statesNumber;

    //! number of finite actions of the mdp. Should be zero for continuos action spaces
    unsigned int actionsNumber;

    //! number of dimensions of the state space states of the mdp. Should be one for finite state spaces
    unsigned int stateDimensionality;

    //! number of dimensions of the state space states of the mdp. Should be one for finite state spaces
    unsigned int actionDimensionality;

    //! number of dimensions of the reward function
    unsigned int rewardDimensionality;

    //! vector of maximum value of each dimension of the reward function
    arma::vec max_obj; //TODO [IMPORTANT][INTERFACE] possiamo mettere un range? (usato per normalizzazione in matlab, default 1). oppure togliere?

    /*!
     * Writes the struct to stream
     */
    inline void writeToStream(std::ostream& out) const
    {
        out << std::setprecision(OS_PRECISION);
        out << statesNumber << "\t" << actionsNumber << std::endl;
        out << stateDimensionality << "\t" << actionDimensionality << std::endl;
        out << rewardDimensionality << std::endl;
        out << gamma << "\t" << isFiniteHorizon << "\t" << horizon << "\t";
        out << isEpisodic << "\t" << isAverageReward << std::endl;
    }

    /*!
     * Reads the struct from stream
     */
    inline void readFromStream(std::istream& in)
    {
        in >> statesNumber >> actionsNumber;
        in >> stateDimensionality >> actionDimensionality;
        in >> rewardDimensionality;
        in >> gamma >> isFiniteHorizon >> horizon;
        in >> isEpisodic >> isAverageReward;
    }
};

/*!
 * The basic interface to log data from Agents.
 * In contains some common data and basic methods to process generic data.
 */
class AgentOutputData
{
public:
    /*!
     * Constructor
     * \param final whether the data logged comes from the end of a run of the algorithm
     */
    AgentOutputData(bool final = false) : final(final), step(0)
    {

    }

    /*!
     * Basic method to write plain data.
     * \param os output stream in which the data should be logged
     */
    virtual void writeData(std::ostream& os) = 0;

    /*!
     * Basic method to write decorated data, e.g. for printing on screen.
     * \param os output stream in which the data should be logged
     */
    virtual void writeDecoratedData(std::ostream& os) = 0;

    /*!
     * Destructor
     */
    virtual ~AgentOutputData()
    {
    }

    /*!
     * Getter.
     * \return if the data logged comes from the end of the run or not.
     */
    inline bool isFinal() const
    {
        return final;
    }

    /*!
     * Getter.
     * \return the step at which the data was logged.
     */
    inline unsigned int getStep() const
    {
        return step;
    }

    /*!
     * Setter. Sets the data step number. Should be used only by the Core.
     */
    inline void setStep(unsigned int step)
    {
        this->step = step;
    }

private:
    unsigned int step;
    bool final;

};

/*!
 * Abstract class for all actions
 */
class Action
{
public:
    /*!
     * writes the action as string. Should be overridden
     */
    inline virtual std::string to_str() const
    {
        return "action";
    }

    /*!
     * Getter.
     * Should be overridden.
     * \return the size of the action if serialized as a csv string
     */
    virtual inline int serializedSize()
    {
        return 0;
    }

    /*!
     * Destructor
     */
    virtual ~Action()
    {

    }

    /*!
     * factory method that transform a set of strings into an action.
     * \param begin the iterator of the first element in the range
     * \param end the iterator to the end of the range
     * \return the action build from strings
     */
    static inline Action generate(std::vector<std::string>::iterator& begin,
                                  std::vector<std::string>::iterator& end)
    {
        return Action();
    }

};

/*!
 * Finite action class.
 * Actions of this types are described by a finite subset of unsigned integers.
 */
class FiniteAction: public Action
{
public:
    /*!
     * Constructor
     * \param n the action number
     */
    FiniteAction(unsigned int n = 0)
    {
        actionN = n;
    }

    /*!
     * This operator is used to convert the action class to an integer
     * \return a reference to the action number
     */
    operator unsigned int&()
    {
        return actionN;
    }

    /*!
     * This operator is used to convert the action class to an integer
     * \return a const reference to the action number
     */
    operator const unsigned int&() const
    {
        return actionN;
    }

    /*!
     * Getter.
     * \return the action number
     */
    inline unsigned int getActionN() const
    {
        return actionN;
    }

    /*!
     * Setter.
     * \param actionN the action number to be set
     */
    inline void setActionN(unsigned int actionN)
    {
        this->actionN = actionN;
    }

    inline virtual std::string to_str() const override
    {
        return std::to_string(actionN);
    }

    inline int serializedSize() override
    {
        return 1;
    }

    /*!
     * Destructor.
     */
    virtual ~FiniteAction()
    {

    }

    /*!
     * \copydoc Action::generate
     */
    static inline FiniteAction generate(std::vector<std::string>::iterator& begin,
                                        std::vector<std::string>::iterator& end)
    {
        int actionN = std::stoul(*begin);
        return FiniteAction(actionN);
    }

    /*!
     * factory method that builds the first N actions
     * \param actionN the number of actions to be build
     * \return all the actions from 0 to actionN-1
     */
    inline static std::vector<FiniteAction> generate(size_t actionN)
    {
        std::vector<FiniteAction> actions;
        for(int i = 0; i < actionN; ++i)
            actions.push_back(FiniteAction(i));
        return actions;
    }

private:
    unsigned int actionN;
};


/*!
 * Dense action class.
 * This class represent an action \f$u\f$, such that \f$u\in\mathbb{R^n}\f$, where n is the dimensionality of action space
 */
class DenseAction: public Action, public arma::vec
{
public:
    /*!
     * Constructor.
     * Builds an action with zero dimensions
     */
    DenseAction ()
    { }

    /*!
     * Constructor.
     * \param size the action dimensionality
     */
    DenseAction(std::size_t size) :
        arma::vec(size)
    {    }

    /*!
     * Constructor.
     * Initialize the action from an armadillo vector
     * \param other the vector to copy as action
     */
    DenseAction(arma::vec& other) :
        arma::vec(other.n_elem)
    {
        this->set_size(other.n_elem);
        for (int i = 0; i < other.n_elem; ++i)
            this->at(i) = other[i];
    }

    inline virtual std::string to_str() const override
    {
        const arma::vec& self = *this;
        std::stringstream ss;
        ss << std::setprecision(OS_PRECISION);

        size_t i;
        for (i = 0; i + 1 < self.n_elem; i++)
            ss << self[i] << ",";

        ss << self[i];

        return ss.str();
    }

    inline int serializedSize() override
    {
        return this->n_elem;
    }

    /*!
     * Destructor.
     */
    virtual ~DenseAction()
    {

    }

    /*!
     * Check whether two actions are almost equals
     * \param other the action to chek againts
     * \param epsilon the tolerance for each element
     * \return whether all elements differences are below the tolerance
     */
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

    /*!
     * \copydoc Action::generate
     */
    static inline DenseAction generate(std::vector<std::string>::iterator& begin,
                                       std::vector<std::string>::iterator& end)
    {
        int size = std::distance(begin, end);
        DenseAction action(size);
        for(unsigned int i = 0; i < size; i++)
        {
            action[i] = std::stod(*(begin +i));
        }

        return action;
    }
};


/*!
 * Abstract class for all states
 */
class State
{
public:
    /*!
     * Constructor.
     */
    State() :
        absorbing(false)
    {

    }

    /*!
     * Getter.
     * \return whether this state is an absorbing one
     */
    inline bool isAbsorbing() const
    {
        return absorbing;
    }

    /*!
     * Setter.
     * \param absorbing whether to set this state as an absorbing state or not.
     */
    inline void setAbsorbing(bool absorbing = true)
    {
        this->absorbing = absorbing;
    }

    /*!
     * writes the state as string. Should be overridden
     */
    inline virtual std::string to_str() const
    {
        return "state";
    }

    /*!
     * Getter.
     * Should be overridden.
     * \return the size of the state if serialized as a csv string
     */
    virtual inline int serializedSize()
    {
        return 0;
    }

    virtual ~State()
    {

    }

    static inline State generate(std::vector<std::string>::iterator& begin,
                                 std::vector<std::string>::iterator& end)
    {
        return State();
    }

private:
    bool absorbing;
};

/*!
 * Finite state class.
 * States of this types are described by a finite subset of unsigned integers.
 */
class FiniteState: public State
{
public:
    /*!
     * Constructor
     * \param n the state number
     */
    FiniteState(unsigned int n = 0)
    {
        stateN = n;
    }

    /*!
     * This operator is used to convert the state class to an integer
     * \return a reference to the state number
     */
    operator size_t&()
    {
        return stateN;
    }

    /*!
     * This operator is used to convert the state class to an integer
     * \return a const reference to the state number
     */
    operator const size_t&() const
    {
        return stateN;
    }

    /*!
     * Getter.
     * \return the action number
     */
    inline std::size_t getStateN() const
    {
        return stateN;
    }

    /*!
     * Setter.
     * \param stateN the action number to be set
     */
    inline void setStateN(std::size_t stateN)
    {
        this->stateN = stateN;
    }

    inline virtual std::string to_str() const override
    {
        return std::to_string(stateN);
    }

    inline int serializedSize() override
    {
        return 1;
    }

    /*!
     * Destructor.
     */
    virtual ~FiniteState()
    {

    }

    /*!
     * \copydoc State::generate
     */
    static inline FiniteState generate(std::vector<std::string>::iterator& begin,
                                       std::vector<std::string>::iterator& end)
    {
        int stateN = std::stoul(*begin);
        return FiniteState(stateN);
    }

private:
    std::size_t stateN;

};

/*!
 * Dense action class.
 * This class represent a state \f$x\f$, such that \f$x\in\mathbb{R^n}\f$, where n is the dimensionality of state space
 */
class DenseState: public State, public arma::vec
{
public:
    /*!
     * Constructor.
     * Builds a state with zero dimensions
     */
    DenseState()
    {
    }

    /*!
     * Constructor.
     * \param size the state dimensionality
     */
    DenseState(std::size_t size) :
        arma::vec(size)
    {
    }

    inline virtual std::string to_str() const override
    {
        const arma::vec& self = *this;
        std::stringstream ss;
        ss << std::setprecision(OS_PRECISION);

        size_t i;
        for (i = 0; i + 1 < self.n_elem; i++)
            ss << self[i] << ",";

        ss << self[i];

        return ss.str();
    }

    inline int serializedSize() override
    {
        return this->n_elem;
    }

    /*!
     * Destructor.
     */
    virtual ~DenseState()
    {

    }

    /*!
     * \copydoc State::generate
     */
    static inline DenseState generate(std::vector<std::string>::iterator& begin,
                                      std::vector<std::string>::iterator& end)
    {
        int size = std::distance(begin, end);
        DenseState state(size);
        for(unsigned int i = 0; i < size; i++)
        {
            state[i] = std::stod(*(begin +i));
        }

        return state;
    }

};

/*!
 * function to generate a reward instance from strings
 */
typedef std::vector<double> Reward;
inline Reward generate(std::vector<std::string>::iterator& begin,
                       std::vector<std::string>::iterator& end)
{
    Reward reward;

    for(auto it = begin; it != end; it++)
    {
        double value = std::stod(*it);
        reward.push_back(value);
    }

    return reward;

}

/*!
 * Operator to write an action to stream
 * \param action the action to be written
 */
inline std::ostream& operator<<(std::ostream& os, const Action& action)
{
    os << action.to_str();
    return os;
}

/*!
 * Operator to write a state to stream
 * \param state the state to be written
 */
inline std::ostream& operator<<(std::ostream& os, const State& state)
{
    os << state.to_str();
    return os;
}

/*!
 * Operator to write the reward to stream
 * \param reward the reward to be written
 */
inline std::ostream& operator<<(std::ostream& os, const Reward& reward)
{
    if(!reward.empty())
    {
        size_t i;
        for (i = 0; i + 1 < reward.size(); i++)
            os << reward[i] << ",";

        os << reward[i];
    }
    return os;
}

}

#endif /* BASICS_H_ */
