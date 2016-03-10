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

#ifndef INCLUDE_RELE_ALGORITHMS_TD_R_LEARNING_H_
#define INCLUDE_RELE_ALGORITHMS_TD_R_LEARNING_H_

#include "TD.h"

namespace ReLe
{

class R_LearningOutput : public FiniteTDOutput
{
public:
    R_LearningOutput(const std::string& alpha,
                     const std::string& beta,
                     const std::string& policyName,
                     const hyperparameters_map& policyHPar,
                     const arma::mat& Q,
                     double ro);

    virtual void writeData(std::ostream& os) override;
    virtual void writeDecoratedData(std::ostream& os) override;

private:
    std::string beta;
    double ro;
};

class R_Learning: public FiniteTD
{
public:
    R_Learning(ActionValuePolicy<FiniteState>& policy, LearningRate& alpha, LearningRate& beta);
    virtual void initEpisode(const FiniteState& state, FiniteAction& action) override;
    virtual void sampleAction(const FiniteState& state, FiniteAction& action) override;
    virtual void step(const Reward& reward, const FiniteState& nextState,
                      FiniteAction& action) override;
    virtual void endEpisode(const Reward& reward) override;

    inline virtual AgentOutputData* getAgentOutputDataEnd() override
    {
        return new R_LearningOutput(alpha.print(), beta.print(), policy.getPolicyName(),
                                    policy.getPolicyHyperparameters(), Q, ro);
    }


    virtual ~R_Learning();


private:
    LearningRate& beta;

    double ro;


};

}


#endif /* INCLUDE_RELE_ALGORITHMS_TD_R_LEARNING_H_ */
