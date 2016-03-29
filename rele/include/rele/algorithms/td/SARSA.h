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

#ifndef INCLUDE_ALGORITHMS_TD_SARSA_H_
#define INCLUDE_ALGORITHMS_TD_SARSA_H_

#include "rele/algorithms/td/TD.h"
#include <armadillo>

namespace ReLe
{

/*!
 * This class implements the SARSA algorithm.
 * This algorithm is an on-policy temporal difference algorithm.
 * Can only work on Finite MDP, i.e. with finite action and state space.
 *
 * References
 * ==========
 *
 * [Rummery, Niranjan. On-line Q-learning using connectionist systems](http://mi.eng.cam.ac.uk/reports/svr-ftp/auto-pdf/rummery_tr166.pdf)
 *
 * [Sutton, Barto. Reinforcement Learning: An Introduction (chapter 6.4)](http://webdocs.cs.ualberta.ca/~sutton/book/ebook/node64.html)
 *
 */
class SARSA: public FiniteTD
{
public:
    /*!
     * Constructor.
     * \param policy the policy to be used by the algorithm
     * \param alpha the learning rate to be used by the algorithm
     */
    SARSA(ActionValuePolicy<FiniteState>& policy, LearningRate& alpha);
    virtual void initEpisode(const FiniteState& state, FiniteAction& action) override;
    virtual void sampleAction(const FiniteState& state, FiniteAction& action) override;
    virtual void step(const Reward& reward, const FiniteState& nextState,
                      FiniteAction& action) override;
    virtual void endEpisode(const Reward& reward) override;

    virtual ~SARSA();


};

/*!
 * This class implements the SARSA(\f$\lambda\f$) algorithm.
 * Differently from the SARSA algorithm, this can use eligibility trace.
 * With \f$\lambda=0\f$, this algorithm is equivalent to plain SARSA, but is
 * less efficient, as it stores the eligibility vector.
 * This algorithm is an on-policy temporal difference algorithm.
 * Can only work on Finite MDP, i.e. with finite action and state space.
 *
 * References
 * ==========
 *
 * [Rummery, Niranjan. On-line Q-learning using connectionist systems](http://mi.eng.cam.ac.uk/reports/svr-ftp/auto-pdf/rummery_tr166.pdf)
 *
 * [Sutton, Barto. Reinforcement Learning: An Introduction (chapter 7.5)](http://webdocs.cs.ualberta.ca/~sutton/book/ebook/node77.html)
 *
 */
class SARSA_lambda: public FiniteTD
{
public:
    /*!
     * Constructor.
     * \param policy the policy to be used by the algorithm
     * \param alpha the learning rate to be used by the algorithm
     * \param accumulating whether to use accumulating trace or replacing ones.
     */
    SARSA_lambda(ActionValuePolicy<FiniteState>& policy, LearningRate& alpha, bool accumulating);
    virtual void initEpisode(const FiniteState& state, FiniteAction& action) override;
    virtual void sampleAction(const FiniteState& state, FiniteAction& action) override;
    virtual void step(const Reward& reward, const FiniteState& nextState,
                      FiniteAction& action) override;
    virtual void endEpisode(const Reward& reward) override;

    virtual ~SARSA_lambda();

    inline void setLambda(double lambda)
    {
        this->lambda = lambda;
    }

protected:
    virtual void init() override;

private:
    double lambda;
    arma::mat Z;
    const bool accumulating;
};

}

#endif /* INCLUDE_ALGORITHMS_TD_SARSA_H_ */
