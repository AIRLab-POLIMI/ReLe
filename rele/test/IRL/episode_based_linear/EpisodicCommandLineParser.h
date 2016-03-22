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

#ifndef TEST_IRL_EPISODE_BASED_LINEAR_EPISODICCOMMANDLINEPARSER_H_
#define TEST_IRL_EPISODE_BASED_LINEAR_EPISODICCOMMANDLINEPARSER_H_

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <iostream>
#include <string>

#include "rele/IRL/utils/IrlGradType.h"


#include "rele/IRL/algorithms/EGIRL.h"
#include "rele/IRL/algorithms/EMIRL.h"
#include "rele/IRL/algorithms/EpisodicExpectedDeltaIRL.h"

namespace ReLe
{

struct irlEpConfig
{
    std::string algorithm;
    IrlEpGrad gradient;
    IrlEpHess hessian;
    int episodes;

    friend std::ostream& operator<< (std::ostream& stream, const irlEpConfig& config)
    {
        stream << "algorithm: " << config.algorithm << std::endl;
        stream << "gradient: " << config.gradient << std::endl;
        stream << "hessian: " << config.hessian << std::endl;
        stream << "episodes: " << config.episodes << std::endl;
    }
};

class CommandLineParser
{
public:
    CommandLineParser()
    {
        std::string gradientDesc = "set the gradient " + IrlEpGradUtils::getOptions();
        std::string hessianDesc = "set the hessian " + IrlEpHessUtils::getOptions();

        desc.add_options() //
        ("help,h", "produce help message") //
        ("algorithm,a", boost::program_options::value<std::string>()->default_value("EMIRL"),
         "set the algorithm (EMIRL | EGIRL)") //
        ("episodes,e", boost::program_options::value<int>()->default_value(1000),
         "set the number of episodes") //
        ("gradient,g", boost::program_options::value<std::string>()->default_value("PGPE_BASELINE"),
         gradientDesc.c_str()) //
        ("hessian,H", boost::program_options::value<std::string>()->default_value("PGPE_BASELINE"),
         hessianDesc.c_str());
    }

    inline irlEpConfig getConfig(int argc, char **argv)
    {
        try
        {
            store(parse_command_line(argc, argv, desc), vm);
            if(vm.count("help"))
            {
                std::cout << desc << std::endl;
                exit(0);
            }
            notify(vm);
        }
        catch (boost::program_options::error& e)
        {
            std::cout << e.what() << std::endl;
            std::cout << desc << std::endl;
            exit(0);
        }

        if(!validate())
        {
            std::cout << "ERROR: wrong parameters. Type --help to get information about parameters.\n";
            exit(0);
        }

        config.algorithm = vm["algorithm"].as<std::string>();
        config.gradient = IrlEpGradUtils::fromString(vm["gradient"].as<std::string>());
        config.hessian = IrlEpHessUtils::fromString(vm["hessian"].as<std::string>());
        config.episodes = vm["episodes"].as<int>();


        std::cout << config << std::endl;

        return config;
    }

private:
    inline bool validate()
    {
        if(vm["episodes"].as<int>() < 1)
            return false;

        std::string algorithm = vm["algorithm"].as<std::string>();
        if(algorithm != "EGIRL" && algorithm != "EMIRL" && algorithm != "EpisodicExpectedDeltaIRL")
            return false;

        std::string gradientType = vm["gradient"].as<std::string>();
        if(!IrlEpGradUtils::isValid(gradientType))
            return false;

        std::string hessianType = vm["hessian"].as<std::string>();
        if(!IrlEpHessUtils::isValid(hessianType))
            return false;

        return true;
    }

private:
    boost::program_options::options_description desc;
    boost::program_options::variables_map vm;
    irlEpConfig config;
};

template<class ActionC, class StateC>
IRLAlgorithm<ActionC, StateC>* buildEpisodicIRLalg(Dataset<ActionC, StateC>& dataset,
        const arma::mat& theta,
        ParametricNormal& dist,
        LinearApproximator& rewardf, double gamma, irlEpConfig conf)
{
    if(conf.algorithm == "EGIRL")
        return new EGIRL<ActionC,StateC>(dataset, theta, dist, rewardf, gamma, conf.gradient);

    else if(conf.algorithm == "EMIRL")
        return new EMIRL<ActionC,StateC>(dataset, theta, dist, rewardf, gamma);

    else if(conf.algorithm == "EpisodicExpectedDeltaIRL")
        return new EpisodicExpectedDeltaIRL<ActionC,StateC>(dataset, theta, dist, rewardf, gamma,
                conf.gradient, conf.hessian);

    return nullptr;

}

}



#endif /* TEST_IRL_EPISODE_BASED_LINEAR_EPISODICCOMMANDLINEPARSER_H_ */
