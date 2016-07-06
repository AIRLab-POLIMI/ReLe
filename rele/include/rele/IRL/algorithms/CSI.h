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

#ifndef INCLUDE_RELE_IRL_ALGORITHMS_CSI_H_
#define INCLUDE_RELE_IRL_ALGORITHMS_CSI_H_

namespace ReLe
{

template<class StateC>
class CSI : public IRLAlgorithm<FiniteAction, StateC>
{
public:
    CSI(Dataset<FiniteAction, StateC>& data, Dataset<FiniteAction, StateC>& dataR,
        LinearApproximator& rewardf, double gamma, unsigned int nActions, bool heuristic = true) :
        data(data), rewardf(rewardf), gamma(gamma), nActions(nActions), heuristic(heuristic)
    {
        lambda_r = 1e-5;
        epsilon = 1e-3;
        N_final = 20;
        lambda_RCAL = 0;
    }

    virtual void run() override
    {
    	//TODO compute phiAllNext
        arma::mat phiData, phiDataNext, phiAll, phiAllNext;
        computeFeatures(phiData, phiDataNext, phiAll);

        //TODO different dataset for regression
        arma::mat& phiData_r = phiData;
        arma::mat& phiAllNext_r = phiAllNext;

        //TODO different features for regression
        arma::mat& psiData_r = phiData_r;

        arma::vec theta_c = RCAL();
        arma::mat Q_r = phiData_r*theta_c;
        arma::mat Q_r_next = phiAllNext_r*theta_c;
        arma::mat Q_r_max = arma::max(Q_r_next, 1);
        arma::vec data_regression = Q_r-gamma*Q_r_max;

        regression(psiData_r, data_regression);

    }

    virtual ~CSI()
    {

    }

private:
    void computeFeatures(arma::mat& phiData, arma::mat& phiDataNext, arma::mat& phiAll)
    {
        unsigned int nTransitions = data.getTransitionsNumber() - data.size();
        unsigned int nFeatures = rewardf.getParametersSize();

        Features& phi = rewardf.getFeatures();

        phiData.set_size(nTransitions, nFeatures);
        phiDataNext.set_size(nTransitions, nFeatures);
        phiAll.set_size(nTransitions*nActions, nFeatures);

        unsigned int count = 0;
        for(auto& episode : data)
        {
            for(unsigned t = 0; t + 1 < episode.size(); t++)
            {
                auto& tr = episode[t];
                auto& trNext = episode[t+1];
                phiData.row(count) = phi(tr.x, tr.u).t();
                phiDataNext.row(count) = phi(trNext.x, trNext.u).t();

                for (unsigned int u = 0; u < nActions; u++)
                {
                    phiAll.row(count + nTransitions*u) = phi(tr.x, FiniteAction(u)).t();
                }

                count++;
            }
        }
    }

    arma::vec RCAL()
    {
        return arma::vec();
    }

    void regression(const arma::mat& phi, const arma::mat& y)
    {
    	arma::mat A = phi.t()*phi+lambda_r*arma::eye(phi.n_cols, phi.n_cols);

    	arma::vec b = phi.t()*y;

    	arma::vec w = arma::solve(A, b);

    	rewardf.setParameters(w);
    }

private:
    Dataset<FiniteAction, StateC>& data;
    LinearApproximator& rewardf;
    double gamma;
    unsigned int nActions;
    bool heuristic;

    double lambda_r;
    double lambda_RCAL;
    double epsilon;
    unsigned int N_final;

};

}


#endif /* INCLUDE_RELE_IRL_ALGORITHMS_CSI_H_ */
