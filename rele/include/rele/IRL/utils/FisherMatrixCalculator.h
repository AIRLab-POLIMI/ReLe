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

#ifndef INCLUDE_RELE_IRL_UTILS_FISHER_FISHERMATRIXCALCULATOR_H_
#define INCLUDE_RELE_IRL_UTILS_FISHER_FISHERMATRIXCALCULATOR_H_

namespace ReLe
{

template<class ActionC, class StateC>
class FisherMatrixcalculator
{
public:
    static arma::mat computeFisherMatrix(DifferentiablePolicy<ActionC, StateC>& policy,
                                         Dataset<ActionC, StateC>& data)
    {
        int dp  = policy.getParametersSize();
        arma::vec localg;
        arma::mat fisher(dp,dp, arma::fill::zeros);

        int nbEpisodes = data.size();
        for (int i = 0; i < nbEpisodes; ++i)
        {
            //core setup
            int nbSteps = data[i].size();

            //iterate the episode
            for (int t = 0; t < nbSteps; ++t)
            {
                Transition<ActionC, StateC>& tr = data[i][t];
                localg = policy.difflog(tr.x, tr.u);
                fisher += localg * localg.t();
            }

        }
        fisher /= nbEpisodes;

        return fisher;
    }

};




}



#endif /* INCLUDE_RELE_IRL_UTILS_FISHER_FISHERMATRIXCALCULATOR_H_ */
