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
#include <type_traits>
#include <string>

#include "rele/core/BasicFunctions.h"
#include "rele/core/BasicsTraits.h"
#include "rele/approximators/Features.h"
#include "rele/utils/CSV.h"


namespace ReLe
{
/*!
 * This struct represent a single transition of the environment (the SARSA tuple).
 * Implements some convenience setters and serialization functions
 */
template<class ActionC, class StateC>
struct Transition
{
    static_assert(std::is_base_of<Action, ActionC>::value, "Not valid Action class as template parameter");
    static_assert(std::is_base_of<State, StateC>::value, "Not a valid State class as template parameter");

    //! The initial state of the transition
    StateC x;
    //! The performed action
    ActionC u;
    //! The state reached after performing the action
    StateC xn;
    //! The reward achieved in the transition
    Reward r;

    /*!
     * Setter.
     * \param x the initial state
     */
    void init(const StateC& x)
    {
        this->x = x;
    }

    /*!
     * Setter.
     * \param u the action performed
     * \param xn the state reached after the performed action
     * \param r the reward achieved
     */
    void update(const ActionC& u, const StateC& xn, const Reward& r)
    {
        this->u = u;
        this->xn = xn;
        this->r = r;
    }

    /*!
     * This method prints the transition header, that consist of tree numbers, that represent
     * the number of comma separated values needed to represent the state, the action and the reward.
     */
    void printHeader(std::ostream& os)
    {
        os << std::setprecision(OS_PRECISION);
        os << x.serializedSize()  << ","
           << u.serializedSize()  << ","
           << r.size()  << std::endl;
    }


    /*!
     * Print the first part of the transition, with non final/non absorbing flag.
     * This function is meaningful when multiple transitions are serialized
     */
    void print(std::ostream& os)
    {
        os << std::setprecision(OS_PRECISION);
        os << "0,0,"
           << x  << ","
           << u  << ","
           << r  << std::endl;
    }

    /*!
     * Print the last part of the transition, with final flag ad appropriate absorbing flag.
     * This function is meaningful when multiple transitions are serialized
     */
    void printLast(std::ostream& os)
    {
        os << std::setprecision(OS_PRECISION);
        os  << "1,"
            << xn.isAbsorbing() << ","
            << xn << std::endl;
    }
};

/*!
 * This class is the collection of multiple transition of a single episode
 * The transition stored in this class should be a meaningful sequence, i.e. the next state
 * of the previous transition should be the initial state of the subsequent one.
 * This class contains some utility functions to perform common operations over an episode.
 */
template <class ActionC, class StateC>
class Episode : public std::vector<Transition<ActionC,StateC>>
{

public:
    /*!
     * This method can be used to compute episode features expectation over transitions.
     * \param phi the features \f$ \phi(x, u, x_n) \f$ to be used
     * \param gamma the discount factor for the features expectations
     * \return a matrix of features expectation, with size phi.rows()\f$\times\f$phi.cols()
     */
    arma::mat computefeatureExpectation(Features& phi, double gamma = 1)
    {
        arma::mat featureExpectation(phi.rows(), phi.cols(), arma::fill::zeros);

        double df = 1;

        Episode& episode = *this;

        for(unsigned int t = 0; t < episode.size(); t++)
        {
            Transition<ActionC, StateC>& transition = episode[t];
            featureExpectation += df * phi(vectorize(transition.x, transition.u, transition.xn));
            df *= gamma;
        }

        return featureExpectation;
    }

    /*!
     * This method returns the episode expected reward
     * \param gamma the discount factor to be used
     * \return the expected reward using gamma as discount factor
     */
    arma::vec getEpisodeReward(double gamma)
    {
        auto& episode = *this;
        unsigned int rewardSize = getRewardSize();
        arma::vec reward(rewardSize, arma::fill::zeros);

        double df = 1.0;
        for(auto& tr : episode)
        {
            for(unsigned int i = 0; i < rewardSize; i++)
                reward(i) += df*tr.r[i];
            df *= gamma;
        }

        return reward;

    }

    /*!
     * Getter.
     * \return the reward size
     */
    unsigned int getRewardSize()
    {
        auto& episode = *this;
        return episode[0].r.size();
    }

    /*!
     * Print the trasition header
     * \see Transition::printHeader
     */
    void printHeader(std::ostream& os)
    {
        os << std::setprecision(OS_PRECISION);
        if(this->size() > 0)
            this->back().printHeader(os);
    }

    /*!
     * Print the dataset to stream
     */
    void print(std::ostream& os)
    {
        os << std::setprecision(OS_PRECISION);
        for(auto& sample : *this)
        {
            sample.print(os);
        }

        this->back().printLast(os);
    }

};

/*!
 * This class represents a dataset, a set of episodes.
 * This class contains some utility functions to perform common operations over a dataset.
 */
template<class ActionC, class StateC>
class Dataset : public std::vector<Episode<ActionC,StateC>>
{

public:
    /*!
     * Computes the mean features expectations over the episodes
     * \param phi the features \f$ \phi(x, u, x_n) \f$ to be used
     * \param gamma the discount factor for the features expectations
     * \return a matrix of features expectation, with size phi.rows()\f$\times\f$phi.cols()
     */
    arma::mat computefeatureExpectation(Features& phi, double gamma = 1)
    {
        size_t episodes = this->size();
        arma::mat featureExpectation(phi.rows(), phi.cols(), arma::fill::zeros);

        for(auto& episode : *this)
        {
            featureExpectation += episode.computefeatureExpectation(phi, gamma);
        }

        featureExpectation /= episodes;

        return featureExpectation;
    }

    /*!
     * Computes the features expectations over the episodes
     * \param phi the features \f$ \phi(x, u, x_n) \f$ to be used
     * \param gamma the discount factor for the features expectations
     * \return a matrix of features expectation, with size (phi.rows()\f$*\f$phi.cols())\f$\times\f$this->size()
     */
    arma::mat computeEpisodeFeatureExpectation(Features& phi, double gamma = 1)
    {
        auto& dataset = *this;
        arma::mat episodeFeatures(phi.rows(), dataset.size());
        bool vectorize = phi.cols() > 1;

        for(unsigned int i = 0; i < dataset.size(); i++)
        {
            if(vectorize)
                episodeFeatures.col(i) = arma::vectorise(dataset[i].computefeatureExpectation(phi, gamma));
            else
                episodeFeatures.col(i) = dataset[i].computefeatureExpectation(phi, gamma);
        }

        return episodeFeatures;
    }

    /*!
     * Getter.
     * \return the total number of transitions contained in this dataset
     */
    unsigned int getTransitionsNumber()
    {
        unsigned int nSamples = 0;

        auto& dataset = *this;
        for(auto& episode : dataset)
            nSamples += episode.size();

        return nSamples;
    }

    /*!
     * Getter.
     * \return the reward size of this dataset
     */
    unsigned int getRewardSize()
    {
        auto& dataset = *this;
        return dataset[0].getRewardSize();
    }

    /*!
     * Computes the episode discounted reward
     * \param gamma the discount factor to be used
     * \return a matrix of the discounted reward, with size this->getRewardSize()\f$\times\f$this->size()
     */
    arma::mat getEpisodesReward(double gamma)
    {
        auto& dataset = *this;

        unsigned int rewardSize = getRewardSize();
        unsigned int episodeN = dataset.size();
        arma::mat rewards(rewardSize, episodeN, arma::fill::zeros);

        unsigned int idx = 0;

        for(auto& episode : dataset)
        {
            rewards.col(idx) = episode.getEpisodeReward(gamma);
            idx++;
        }

        return rewards;
    }

    /*!
     * Computes the mean discounted reward.
     * \param gamma the discount factor to be used
     * \return a vector of the mean discounted reward, with size this->getRewardSize()
     */
    arma::vec getMeanReward(double gamma)
    {
        return arma::mean(this->getEpisodesReward(gamma), 1);
    }

    /*!
     * Get all transitions rewards as matrix
     * \return a matrix of the transition rewards, with size this->getRewardSize()\f$\times\f$this->getTransitionsNumber()
     */
    arma::mat rewardAsMatrix()
    {
        auto& dataset = *this;

        unsigned int rewardSize = getRewardSize();
        unsigned int nTransitions = getTransitionsNumber();
        arma::mat rewards(rewardSize, nTransitions);

        unsigned int idx = 0;

        for(auto& episode : dataset)
        {
            for(auto& tr : episode)
            {
                for(unsigned int i = 0; i < rewardSize; i++)
                    rewards(i, idx) = tr.r[i];
                idx++;
            }
        }

        return rewards;
    }

    /*!
     * Get all features over transitions as matrix
     * \param phi the features \f$ \phi(x, u, x_n) \f$ to be used
     * \return a matrix of the transition features, with size (phi.rows()\f$*\f$phi.cols())\f$\times\f$this->getTransitionsNumber()
     */
    arma::mat featuresAsMatrix(Features& phi)
    {
        auto& dataset = *this;

        unsigned int featuresSize = phi.rows();
        unsigned int nTransitions = getTransitionsNumber();
        arma::mat features(featuresSize, nTransitions);

        unsigned int idx = 0;

        for(auto& episode : dataset)
        {
            for(auto& tr : episode)
            {
                features.col(idx) = phi(tr.x, tr.u, tr.xn);
                idx++;
            }
        }

        return features;
    }

    /*!
     * This method joins two datasets
     * \param data the new dataset to join to the prevoious one.
     */
    void addData(Dataset<ActionC, StateC>& data)
    {
        this->insert(this->data.end(), data.begin(), data.end());
    }

    /*!
     * Setter.
     * \param data the new dataset to be set
     */
    void setData(Dataset<ActionC, StateC>& data)
    {
        this->erase();
        addData(data);
    }

    /*!
     * Getter.
     * \return the number of episodes in this dataset
     */
    unsigned int getEpisodesNumber()
    {
        return this->size();
    }

    /*!
     * Getter.
     * \return the maximum episode length
     */
    unsigned int getEpisodeMaxLength()
    {
        unsigned int max = 0;
        for(auto& episode : *this)
        {
            unsigned int steps = episode.size();
            max = std::max(steps, max);
        }

        return max;
    }


public:
    /*!
     * This method write an episode to an output stream
     * \param os the output stream
     */
    void writeToStream(std::ostream& os)
    {
        os << std::setprecision(OS_PRECISION);
        if (this->size() > 0)
        {
            this->back().printHeader(os);

            for (auto& episode : *this)
            {
                episode.print(os);
            }
        }
    }

    /*!
     * This method read an episode from an input stream
     * \param is the input stream
     */
    void readFromStream(std::istream& is)
    {
        std::vector<std::string> header;
        CSVutils::readCSVLine(is, header);

        int stateSize = std::stoi(header[0]);
        int actionSize = std::stoi(header[1]);
        int rewardSize = std::stoi(header[2]);

        while(is)
        {
            readEpisodeFromStream(is, stateSize, actionSize, rewardSize);
        }

    }

private:
    void readEpisodeFromStream(std::istream& is, int stateSize, int actionSize,
                               int rewardSize)
    {
        bool first = true;
        bool last = false;
        Episode<ActionC, StateC> episode;
        Transition<ActionC, StateC> transition;

        std::vector<std::string> line;
        while (!last && CSVutils::readCSVLine(is, line))
        {
            last = std::stoi(line[0]);
            bool absorbing = std::stoi(line[1]);

            auto sit = line.begin() + 2;
            auto ait = sit + stateSize;
            auto rit = ait + actionSize;
            auto endit = rit + rewardSize;

            StateC state = StateC::generate(sit, ait);
            state.setAbsorbing(absorbing);

            if (!first)
            {
                transition.xn = state;
                episode.push_back(transition);
            }

            if (!last)
            {
                transition.x = state;
                transition.u = ActionC::generate(ait, rit);
                transition.r = generate(rit, endit);
            }

            line.clear();
            first = false;
        }

        //Add episode to dataset
        if(last)
            this->push_back(episode);
    }

};


}

#endif /* INCLUDE_RELE_CORE_TRANSITION_H_ */
