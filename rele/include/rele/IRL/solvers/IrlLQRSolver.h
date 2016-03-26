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

#ifndef INCLUDE_RELE_IRL_SOLVERS_IRLLQRSOLVER_H_
#define INCLUDE_RELE_IRL_SOLVERS_IRLLQRSOLVER_H_

#include "rele/solvers/lqr/LQRsolver.h"
#include "rele/IRL/IRLSolver.h"

namespace ReLe
{

class IRL_LQRSolver : public LQRsolver, public IRLSolver<DenseAction, DenseState>
{
public:
    IRL_LQRSolver(LQR& lqr, Features& phi, Features& phiPi)
        : IRLSolver<DenseAction, DenseState>(phi), LQRsolver(lqr, phiPi, Type::MOO)
    {

    }

    void solve() override
    {
        LQRsolver::solve();
    }

    virtual inline Dataset<DenseAction, DenseState> test() override
    {
        return LQRsolver::test();
    }

    inline void setWeights(arma::vec& weights) override
    {
        LQRsolver::setRewardWeights(weights);
    }

    inline unsigned int getBasisSize()
    {
        return phi.rows();
    }

    inline double getGamma() override
    {
        return gamma;
    }

    inline Policy<DenseAction, DenseState>& getPolicy() override
    {
        LQRsolver::getPolicy();
    }

    inline void setTestParams(unsigned int testEpisodes,
                              unsigned int testEpisodeLength)
    {
        LQRsolver::setTestParams(testEpisodes, testEpisodeLength);
    }

    inline arma::mat computeFeaturesExpectations() override
    {
        Dataset<DenseAction, DenseState>&& dataset = LQRsolver::test();
        return dataset.computefeatureExpectation(phi, gamma);
    }

};

}

#endif /* INCLUDE_RELE_IRL_SOLVERS_IRLLQRSOLVER_H_ */
