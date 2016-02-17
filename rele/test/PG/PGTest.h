/*
 * rele,
 *
 *
 * Copyright (C) 2015  Davide Tateo & Matteo Pirotta
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

#include "rele/algorithms/policy_search/gradient/onpolicy/REINFORCEAlgorithm.h"
#include "rele/algorithms/policy_search/gradient/onpolicy/GPOMDPAlgorithm.h"
#include "rele/algorithms/policy_search/gradient/onpolicy/NaturalPGAlgorithm.h"
#include "rele/algorithms/policy_search/gradient/onpolicy/ENACAlgorithm.h"
#include "rele/core/Core.h"

#include "rele/utils/FileManager.h"

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <iostream>
#include <string>

using namespace std;
using namespace ReLe;
using namespace boost::program_options;

struct gradConfig
{
    string envName;
    string algorithm;
    int nbRuns;
    int nbEpisodes;
    double stepLength;
    StepRule* stepRule;
};

class CommandLineParser
{
public:
    CommandLineParser()
    {
        desc.add_options() //
        ("help, h", "produce help message") //
        ("algorithm, a", value<string>()->default_value("r"),
         "set the algorithm") //
        ("updates, u", value<int>()->default_value(400),
         "set the number of updates") //
        ("episodes, e", value<int>()->default_value(100),
         "set the number of episodes") //
        ("stepLength, s", value<double>()->default_value(0.01),
         "set the step length") //
        ("stepRule, r", value<string>()->default_value("adaptive"),
         "set the step rule");
    }

    gradConfig getConfig(int argc, char **argv)
    {
        try
        {
            store(parse_command_line(argc, argv, desc), vm);
            if(vm.count("help"))
            {
                cout << desc << endl;
                exit(0);
            }
            notify(vm);
        }
        catch (error& e)
        {
            cout << e.what() << endl;
            cout << desc << endl;
            exit(0);
        }

        if(!validate())
        {
            cout << "ERROR: wrong parameters. Type --help to get information about parameters.\n";
            exit(0);
        }

        config.algorithm = vm["algorithm"].as<string>();
        config.nbRuns = vm["updates"].as<int>();
        config.nbEpisodes = vm["episodes"].as<int>();
        config.stepLength = vm["stepLength"].as<double>();
        if(vm["stepRule"].as<string>() == "constant")
            config.stepRule = new ConstantStep(config.stepLength);
        else if(vm["stepRule"].as<string>() == "adaptive")
            config.stepRule = new AdaptiveStep(config.stepLength);

        return config;
    }

private:
    boost::program_options::options_description desc;
    boost::program_options::variables_map vm;
    gradConfig config;

private:
    bool validate()
    {
        if(vm["updates"].as<int>() < 1 ||
                vm["episodes"].as<int>() < 1 ||
                vm["stepLength"].as<double>() <= 0)
            return false;

        string stepRule = vm["stepRule"].as<string>();
        if(stepRule != "constant" && stepRule != "adaptive")
            return false;

        string algorithm = vm["algorithm"].as<string>();
        if(algorithm != "r" && algorithm != "rb" && algorithm != "g" && algorithm != "gb"
                && algorithm != "gsb" && algorithm != "natg" && algorithm != "natr"
                && algorithm != "enac")
            return false;

        return true;
    }
};

template<class ActionC, class StateC>
class PGTest
{
public:
    PGTest(gradConfig config,
           Environment<ActionC, StateC>& mdp,
           DifferentiablePolicy<ActionC, StateC>& policy) :
        config(config),
        mdp(mdp),
        policy(policy)
    {
        setAgent();
    }

    ~PGTest()
    {
        delete config.stepRule;
    }

    void run()
    {
        FileManager fm(config.envName, "PG");
        fm.createDir();
        std::cout << std::setprecision(OS_PRECISION);

        auto&& core = buildCore(mdp, *agent);
        core.getSettings().loggerStrategy = new WriteStrategy<ActionC, StateC>(
            fm.addPath(outputName),
            WriteStrategy<ActionC, StateC>::AGENT,
            true /*delete file*/
        );

        int horiz = mdp.getSettings().horizon;
        core.getSettings().episodeLength = horiz;

        int nbepperpol = config.nbEpisodes;
        int nbUpdates = config.nbRuns;
        int episodes  = nbUpdates * nbepperpol;
        double every, bevery;
        every = bevery = 0.1; //%
        int updateCount = 0;
        for (int i = 0; i < episodes; i++)
        {
            if (i % nbepperpol == 0)
            {
                updateCount++;
                if ((updateCount >= nbUpdates*every) || (updateCount == 1))
                {
                    int p = std::floor(100 * (updateCount/static_cast<double>(nbUpdates)));
                    cout << "### " << p << "% ###" << endl;
                    cout << policy.getParameters().t();
                    core.getSettings().testEpisodeN = 1000;
                    arma::vec J = core.runBatchTest();
                    cout << "mean score: " << J(0) << endl;
                    if (updateCount != 1)
                        every += bevery;
                }
            }

            core.runEpisode();
        }
    }

protected:
    gradConfig config;
    string outputName;
    Environment<ActionC, StateC>& mdp;
    AbstractPolicyGradientAlgorithm<ActionC, StateC>* agent;
    DifferentiablePolicy<ActionC, StateC>& policy;

protected:
    void setAgent()
    {
        int nbepperpol = config.nbEpisodes;
        unsigned int rewardId = 0;
        if(config.algorithm == "r")
        {
            cout << "REINFORCEAlgorithm" << endl;
            bool usebaseline = false;
            agent = new REINFORCEAlgorithm<ActionC, StateC>(policy, nbepperpol,
                    *(config.stepRule), usebaseline, rewardId);
            outputName = config.envName + "_r.log";
        }
        else if(config.algorithm == "g")
        {
            cout << "GPOMDPAlgorithm" << endl;
            agent = new GPOMDPAlgorithm<ActionC, StateC>(policy, nbepperpol,
                    mdp.getSettings().horizon, *(config.stepRule), rewardId);
            outputName = config.envName + "_g.log";
        }
        else if(config.algorithm == "rb")
        {
            cout << "REINFORCEAlgorithm BASELINE" << endl;
            bool usebaseline = true;
            agent = new REINFORCEAlgorithm<ActionC, StateC>(policy, nbepperpol,
                    *(config.stepRule), usebaseline, rewardId);
            outputName = config.envName + "_rb.log";
        }
        else if(config.algorithm == "gb")
        {
            cout << "GPOMDPAlgorithm BASELINE" << endl;
            agent = new GPOMDPAlgorithm<ActionC, StateC>(policy, nbepperpol,
                    mdp.getSettings().horizon, *(config.stepRule),
                    GPOMDPAlgorithm<ActionC, StateC>::BaseLineType::MULTI,
                    rewardId);
            outputName = config.envName + "_gb.log";
        }
        else if(config.algorithm == "gsb")
        {
            cout << "GPOMDPAlgorithm SINGLE BASELINE" << endl;
            agent = new GPOMDPAlgorithm<ActionC, StateC>(policy, nbepperpol,
                    mdp.getSettings().horizon, *(config.stepRule),
                    GPOMDPAlgorithm<ActionC, StateC>::BaseLineType::SINGLE,
                    rewardId);
            outputName = config.envName + "_gsb.log";
        }
        else if(config.algorithm == "natg")
        {
            cout << "NaturalGPOMDPAlgorithm BASELINE" << endl;
            bool usebaseline = true;
            agent = new NaturalGPOMDPAlgorithm<ActionC, StateC>(policy, nbepperpol,
                    mdp.getSettings().horizon, *(config.stepRule), usebaseline, rewardId);
            outputName = config.envName + "_natg.log";
        }
        else if(config.algorithm == "natr")
        {
            cout << "NaturalREINFORCEAlgorithm BASELINE" << endl;
            bool usebaseline = true;
            agent = new NaturalREINFORCEAlgorithm<ActionC, StateC>(policy, nbepperpol,
                    *(config.stepRule), usebaseline, rewardId);
            outputName = config.envName + "_natr.log";
        }
        else if (config.algorithm == "enac")
        {
            cout << "eNAC BASELINE" << endl;
            bool usebaseline = true;
            agent = new eNACAlgorithm<ActionC, StateC>(policy, nbepperpol,
                    *(config.stepRule), usebaseline, rewardId);
            outputName = config.envName + "_enac.log";
        }
    }
};
