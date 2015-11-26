/*
*  rele,
*
*
*  Copyright (C) 2015 Davide Tateo & Matteo Pirotta
*  Versione 1.0
*
*  This file is part of rele.
*
*  rele is free software: you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation, either version 3 of the License, or
*  (at your option) any later version.
*
*  rele is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with rele.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef FQI_H_
#define FQI_H_

namespace ReLe
{
/**
* Fitted Q-Iteration (FQI)
*
* "Tree-Based Batch Mode Reinforcement Learning"
* Damien Ernst, Pierre Geurts, Louis Wehenkel
* Journal of Machine Learning Research, 6, 2006, pp. 503-556.
*/

template<class StateC>
class FQI
{
public:
    FQI(Dataset<FiniteAction, StateC>& data, BatchRegressor& QRegressor,
        unsigned int nActions, double gamma) :
        data(data),
        QRegressor(QRegressor),
        nActions(nActions),
        gamma(gamma)
    {

    }

    void run(Features& phi, unsigned int maxiterations, double epsilon = 0.1)
    {
        /*** Initialize variables ***/
        unsigned int nEpisodes = data.size();
        // compute the overall number of samples
        unsigned int nSamples = 0;
        for(unsigned int k = 0; k < nEpisodes; k++)
            nSamples += data[k].size();

        arma::vec&& rewards = data.rewardAsMAtrix();
        arma::vec nextStates(nSamples, arma::fill::zeros);
        arma::mat input = data.featuresAsMatrix(phi);
        arma::vec output(nSamples, arma::fill::zeros);
        Q.zeros(nSamples, nActions);

        double J;

        // compute Q_(i+1)
        unsigned int iteration = 0;
        /*** Main FQI loop ***/
        do
        {
            arma::vec output;
            // Update and print the number of iterations
            ++iteration;
            std::cout << "*********************************************************" << std::endl;
            std::cout << "FQI iteration: " << iteration << std::endl;

            // Evaluate the current policy using the current approximation
            // of Q functions
            unsigned int i = 0;
            for(auto& episode : data)
            {
                for(auto& tr : episode)
                {
                    arma::vec Q_xn;
                    for(unsigned int u = 0; u < nActions; u++)
                    {
                        Q_xn(u) = arma::as_scalar(QRegressor(tr.xn, FiniteAction(u)));
                    }
                    output(i) = rewards(i) + gamma * arma::max(Q_xn);
                    i++;
                }
            }

            // Use Regressor
            QRegressor.trainFeatures(input, output);
            arma::mat Qprev = Q;
            computeQ(data);
            J = arma::norm(Q - Qprev);

        } while((iteration < maxiterations) && (J > epsilon));

        /*** Display some info ***/
        std::cout << "*********************************************************" << std::endl;
        if(J > epsilon)
            std::cout << "FQI finished in " << iteration <<
                      " iterations WITHOUT CONVERGENCE to a fixed point" << std::endl;
        else
            std::cout << "FQI converged in " << iteration << " iterations" << std::endl <<
                      "********************************************************* " << std::endl;
    }

    void computeQ(Dataset<FiniteAction, StateC>& data)
    {
        unsigned int i = 0;
        for(auto& episode : data)
        {
            for(auto& tr : episode)
            {
                for(unsigned int u = 0; u < nActions; u++)
                {
                    Q(i, u) = arma::as_scalar(QRegressor(tr.x, FiniteAction(u)));
                }
                i++;
            }
        }
    }

protected:
    Dataset<FiniteAction, StateC>& data;
    arma::mat Q;
    BatchRegressor& QRegressor;
    unsigned int nActions;
    double gamma;
};
}

#endif //FQI_H
