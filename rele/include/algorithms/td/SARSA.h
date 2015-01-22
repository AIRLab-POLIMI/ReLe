/*
 * SARSA.h
 *
 *  Created on: 12/gen/2015
 *      Author: dave
 */

#ifndef INCLUDE_ALGORITHMS_TD_SARSA_H_
#define INCLUDE_ALGORITHMS_TD_SARSA_H_

#include "td/TD.h"
#include <armadillo>

namespace ReLe
{

class SARSA: public FiniteTD
{
public:
    SARSA(std::size_t statesN, std::size_t actionN);
    virtual void initEpisode(const FiniteState& state, FiniteAction& action);
    virtual void sampleAction(const FiniteState& state, FiniteAction& action);
    virtual void step(const Reward& reward, const FiniteState& nextState,
                      FiniteAction& action);
    virtual void endEpisode(const Reward& reward);
    virtual void endEpisode();

    virtual ~SARSA();

protected:
    virtual void printStatistics();

};

class SARSA_lambda: public Agent<FiniteAction, FiniteState>
{
public:
    SARSA_lambda(double lambda);
    virtual void initEpisode(const FiniteState& state, FiniteAction& action);
    virtual void sampleAction(const FiniteState& state, FiniteAction& action);
    virtual void step(const Reward& reward, const FiniteState& nextState,
                      FiniteAction& action);
    virtual void endEpisode(const Reward& reward);
    virtual void endEpisode();

    virtual ~SARSA_lambda();

private:
    double lambda;
};

}

#endif /* INCLUDE_ALGORITHMS_TD_SARSA_H_ */
