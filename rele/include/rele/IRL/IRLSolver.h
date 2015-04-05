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

#ifndef INCLUDE_RELE_IRL_IRLSOLVER_H_
#define INCLUDE_RELE_IRL_IRLSOLVER_H_

#include "Solver.h"
#include "ParametricRewardMDP.h"

namespace ReLe
{

template<class ActionC, class StateC>
class IRLSolver : public Solver<ActionC, StateC>
{
public:
    IRLSolver(Environment<ActionC, StateC>& mdp, AbstractBasisMatrix& features, ParametricRegressor& rewardRegressor)
        : prMdp(mdp, rewardRegressor), features(features), rewardRegressor(rewardRegressor)
    {

    }

    inline void setWeights(arma::vec& weights)
    {
        rewardRegressor.setParameters(weights);
    }

    arma::mat computeFeaturesExpectations()
    {
        Dataset<ActionC, StateC>&& data = this->test();
        return data.computefeatureExpectation(features, mdp.getSettings().gamma);
    }

protected:
    ParametricRewardMDP<ActionC, StateC> prMdp;
    AbstractBasisMatrix& features;
    ParametricRegressor& rewardRegressor;

};


/*template<class ActionC, class StateC, class SolverC>
class IRLOracleSolver : public IRLSolver<ActionC, StateC>
{
    static_assert(std::is_base_of<Solver<ActionC, StateC>, SolverC>::value, "Not valid Solver class as template parameter");
public:
    IRLOracleSolver(Solver<ActionC, StateC>& solver, Environment<ActionC, StateC>& mdp,
                    AbstractBasisMatrix& features, ParametricRegressor& rewardRegressor)
        : IRLSolver(mdp, features, rewardRegressor), solver(prMdp)
    {

    }

    virtual Dataset<ActionC, StateC> test()
    {
    	return solver.test();
    }


    virtual Policy<ActionC, StateC>& getPolicy()
    {
    	return solver.getPolicy();
    }

    virtual ~IRLOracleSolver()
    {

    }

private:
    SolverC solver;

};*/


}

#endif /* INCLUDE_RELE_IRL_IRLSOLVER_H_ */
