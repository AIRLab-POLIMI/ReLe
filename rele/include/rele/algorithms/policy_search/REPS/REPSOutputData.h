/*
 * rele,
 *
 *
 * Copyright (C) 2015 Davide Tateo
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

#ifndef INCLUDE_RELE_ALGORITHMS_POLICY_SEARCH_REPS_REPSOUTPUTDATA_H_
#define INCLUDE_RELE_ALGORITHMS_POLICY_SEARCH_REPS_REPSOUTPUTDATA_H_

#include "Basics.h"

#include "policy_search/BlackBoxOutputData.h"

namespace ReLe
{

class AbstractREPSOutputData: public AgentOutputData
{
public:
    AbstractREPSOutputData(int N, double eps, const std::string& policyName, bool final);

    virtual void writeData(std::ostream& os) = 0;
    virtual void writeDecoratedData(std::ostream& os) = 0;

    virtual ~AbstractREPSOutputData();

protected:
    void writeInfo(std::ostream& os);
    void writeDecoratedInfo(std::ostream& os);

private:
    std::string policyName;
    int N;
    double eps;

};

class TabularREPSOutputData: public AbstractREPSOutputData
{
public:
    TabularREPSOutputData(int N, double eps, const std::string& policyPrinted, bool final);

    virtual void writeData(std::ostream& os);
    virtual void writeDecoratedData(std::ostream& os);

    virtual ~TabularREPSOutputData();

private:
    std::string policyPrinted;
};

/*
class EpisodicREPSOutputData: public AgentOutputData
{
public:
    EpisodicREPSOutputData(double eps, const std::string& policyName,
                           const arma::vec& policyParameters,
                           const arma::mat& policyVariance,
                           std::vector<ParameterSample>& samples);

    virtual void writeData(std::ostream& os);
    virtual void writeDecoratedData(std::ostream& os);

    virtual ~EpisodicREPSOutputData();

private:
    double eps;
    std::string policyName;
    arma::vec policyParameters;
    arma::mat policyVariance;

    std::vector<ParameterSample> samples;

};*/

class REPSOutputData : public BlackBoxOutputData<BlackBoxPolicyIndividual>
{
public:
    REPSOutputData(unsigned int nbIndividual, unsigned int nbParams,
                   unsigned int nbEvals);
    virtual void writeData(std::ostream& os);
    virtual void writeDecoratedData(std::ostream& os);

private:

};



}


#endif /* INCLUDE_RELE_ALGORITHMS_POLICY_SEARCH_REPS_REPSOUTPUTDATA_H_ */
