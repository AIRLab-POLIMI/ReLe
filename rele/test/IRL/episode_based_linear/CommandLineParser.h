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

#ifndef TEST_IRL_EPISODE_BASED_LINEAR_COMMANDLINEPARSER_H_
#define TEST_IRL_EPISODE_BASED_LINEAR_COMMANDLINEPARSER_H_

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <iostream>
#include <string>

#include "rele/IRL/utils/IrlGradType.h"


#include "rele/IRL/algorithms/GIRL.h"
#include "rele/IRL/algorithms/PGIRL.h"
#include "rele/IRL/algorithms/ExpectedDeltaIRL.h"

namespace ReLe
{

struct irlConfig
{
    std::string algorithm;
    int episodes;

    friend std::ostream& operator<< (std::ostream& stream, const irlConfig& config)
    {
        stream << "algorithm: " << config.algorithm << std::endl;
        stream << "episodes: " << config.episodes << std::endl;
    }
};

class CommandLineParser
{
public:
	CommandLineParser()
    {
        std::string gradientDesc = "set the gradient " + IrlGradUtils::getOptions();
        std::string hessianDesc = "set the hessian " + IrlHessUtils::getOptions();

        desc.add_options() //
        ("help,h", "produce help message") //
        ("algorithm,a", boost::program_options::value<std::string>()->default_value("EMIRL"),
         "set the algorithm (EMIRL | EGIRL)") //
        ("episodes,e", boost::program_options::value<int>()->default_value(1000),
         "set the number of episodes");
    }

    inline irlConfig getConfig(int argc, char **argv)
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
        if(algorithm != "EGIRL" && algorithm != "EMIRL")
            return false;

        return true;
    }

private:
    boost::program_options::options_description desc;
    boost::program_options::variables_map vm;
    irlConfig config;
};

template<class ActionC, class StateC>
IRLAlgorithm<ActionC, StateC>* buildIRLalg(Dataset<ActionC, StateC>& dataset,
        DifferentiablePolicy<ActionC, StateC>& policy,
        LinearApproximator& rewardf, double gamma, irlConfig conf)
{
    if(conf.algorithm == "EGIRL")
        return new EGIRL<DenseAction,DenseState>(dataset, policy, rewardf,
                                                gamma);
    else if(conf.algorithm == "EMIRL")
        return new EMIRL<DenseAction,DenseState>(dataset, policy, rewardf,
                gamma);



}

}



#endif /* TEST_IRL_EPISODE_BASED_LINEAR_COMMANDLINEPARSER_H_ */
