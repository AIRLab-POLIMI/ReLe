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

#include "rele/core/BasicFunctions.h"
#include "rele/core/BasicsTraits.h"
#include "rele/approximators/Features.h"
#include "rele/utils/CSV.h"


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

    void printHeader(std::ostream& os)
    {
        os << std::setprecision(OS_PRECISION);
        os << x.serializedSize()  << ","
           << u.serializedSize()  << ","
           << r.size()  << std::endl;
    }


    void print(std::ostream& os)
    {
        os << std::setprecision(OS_PRECISION);
        os << "0,0,"
           << x  << ","
           << u  << ","
           << r  << std::endl;
    }

    void printLast(std::ostream& os)
    {
        os << std::setprecision(OS_PRECISION);
        os  << "1,"
            << xn.isAbsorbing() << ","
            << xn << std::endl;
    }
};

template <class ActionC, class StateC>
class Episode : public std::vector<Transition<ActionC,StateC>>
{

public:
    //TODO add template method...
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

    void printHeader(std::ostream& os)
    {
        os << std::setprecision(OS_PRECISION);
        if(this->size() > 0)
            this->back().printHeader(os);
    }


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

template<class ActionC, class StateC>
class Dataset : public std::vector<Episode<ActionC,StateC>>
{

public:

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

    unsigned int getTransitionsNumber()
    {
        unsigned int nSamples = 0;

        auto& dataset = *this;
        for(auto& episode : dataset)
            nSamples += episode.size();

        return nSamples;
    }

    unsigned int getRewardSize()
    {
        auto& dataset = *this;
        return dataset[0][0].r.size();
    }

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

    template<class InputC>
    arma::mat featuresAsMatrix(Features_<InputC>& phi)
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
                features.col(idx) = phi(tr.x, tr.u);
                idx++;
            }
        }

        return features;
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

    unsigned int getEpisodesNumber()
    {
        return this->size();
    }

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
