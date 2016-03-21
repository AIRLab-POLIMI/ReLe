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

#ifndef TEST_IRL_IRLCOMMANDLINEPARSER_H_
#define TEST_IRL_IRLCOMMANDLINEPARSER_H_


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
    IrlGrad gradient;
    IrlHess hessian;
    int episodes;

    friend std::ostream& operator<< (std::ostream& stream, const irlConfig& config)
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
        std::string gradientDesc = "set the gradient " + IrlGradUtils::getOptions();
        std::string hessianDesc = "set the hessian " + IrlHessUtils::getOptions();

        desc.add_options() //
        ("help,h", "produce help message") //
        ("algorithm,a", boost::program_options::value<std::string>()->default_value("GIRL"),
         "set the algorithm (GIRL | PGIRL | ExpectedDeltaIRL)") //
        ("episodes,e", boost::program_options::value<int>()->default_value(1000),
         "set the number of episodes") //
        ("gradient,g", boost::program_options::value<std::string>()->default_value("REINFORCE"),
         gradientDesc.c_str()) //
        ("hessian,H", boost::program_options::value<std::string>()->default_value("REINFORCE"),
         hessianDesc.c_str());
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
        config.gradient = IrlGradUtils::fromString(vm["gradient"].as<std::string>());
        config.hessian = IrlHessUtils::fromString(vm["hessian"].as<std::string>());
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
        if(algorithm != "GIRL" && algorithm != "PGIRL" && algorithm != "ExpectedDeltaIRL")
            return false;

        std::string gradientType = vm["gradient"].as<std::string>();
        if(!IrlGradUtils::isValid(gradientType))
            return false;

        std::string hessianType = vm["hessian"].as<std::string>();
        if(!IrlHessUtils::isValid(hessianType))
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
    if(conf.algorithm == "GIRL")
        return new GIRL<DenseAction,DenseState>(dataset, policy, rewardf,
                                                gamma, conf.gradient);
    else if(conf.algorithm == "PGIRL")
        return new PlaneGIRL<DenseAction,DenseState>(dataset, policy, rewardf,
                gamma, conf.gradient);

    else if(conf.algorithm == "ExpectedDeltaIRL")
        return new ExpectedDeltaIRL<DenseAction,DenseState>(dataset, policy, rewardf,
                gamma, conf.gradient, conf.hessian);



}

}


#endif /* TEST_IRL_IRLCOMMANDLINEPARSER_H_ */
