/*
 * UCB1.h
 *
 *  Created on: Dec 22, 2015
 *      Author: francesco
 */

#ifndef UCB1_H_
#define UCB1_H_

#include "MABAlgorithm/MABAlgorithm.h"

namespace ReLe
{

template<class StateC>
class UCB1: public MABAlgorithm<FiniteAction, StateC>
{
public:
    UCB1(ParametricPolicy<FiniteAction, StateC>& policy):
        MABAlgorithm<FiniteAction, StateC>(policy)
    {
    }

    virtual ~UCB1()
    {
    }

protected:
    virtual arma::vec selectNextArm() override
    {
        return arma::vec();
    }

    virtual void updateHistory(const Reward& reward) override
    {

    }

};

}

#endif /* UCB1_H_ */
