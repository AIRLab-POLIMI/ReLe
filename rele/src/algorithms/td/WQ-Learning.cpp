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

#include "rele/algorithms/td/WQ-Learning.h"

using namespace std;
using namespace arma;

namespace ReLe
{

WQ_Learning::WQ_Learning(ActionValuePolicy<FiniteState>& policy, LearningRate& alpha) :
    Q_Learning(policy, alpha)
{
}

void WQ_Learning::initEpisode(const FiniteState& state, FiniteAction& action)
{
    sampleAction(state, action);
}

void WQ_Learning::sampleAction(const FiniteState& state, FiniteAction& action)
{
    x = state.getStateN();
    u = policy(x);

    action.setActionN(u);
}

void WQ_Learning::step(const Reward& reward, const FiniteState& nextState,
                       FiniteAction& action)
{
    size_t xn = nextState.getStateN();
    double r = reward[0];

    arma::vec integrals(task.actionsNumber, arma::fill::zeros);
    for(unsigned int i = 0; i < integrals.n_elem; i++)
    {
        arma::vec means = Q.row(xn).t();
        arma::vec sigma = sampleStdQ.row(xn).t();
        double pdfMean = means(i);
        double pdfSampleStd = sigma(i);
        double lowerLimit = pdfMean - sigmaBound * pdfSampleStd;
        double upperLimit = pdfMean + sigmaBound * pdfSampleStd;

        arma::vec trapz = arma::linspace(lowerLimit, upperLimit, nTrapz + 1);
        double diff = trapz(1) - trapz(0);

        double result = 0;
        for(unsigned int t = 0; t < trapz.n_elem - 1; t++)
        {
            arma::vec cdfs(idxs.n_cols, arma::fill::zeros);
            for(unsigned int j = 0; j < cdfs.n_elem; j++)
            {
                boost::math::normal cdfNormal(means(idxs(i, j)), sigma(idxs(i, j)));
                cdfs(j) = cdf(cdfNormal, trapz(t));
            }
            boost::math::normal pdfNormal(pdfMean, pdfSampleStd);
            double t1 = pdf(pdfNormal, trapz(t)) * arma::prod(cdfs);

            for(unsigned int j = 0; j < cdfs.n_elem; j++)
            {
                boost::math::normal cdfNormal(means(idxs(i, j)), sigma(idxs(i, j)));
                cdfs(j) = cdf(cdfNormal, trapz(t + 1));
            }
            double t2 = pdf(pdfNormal, trapz(t + 1)) * arma::prod(cdfs);

            result += (t1 + t2) * diff * 0.5;
        }

        integrals(i) = result;
    }

    double W = arma::dot(Q.row(xn), integrals);

    double target = r + task.gamma * W;

    updateMeanAndSampleStdQ(target);

    x = xn;
    u = policy(xn);

    action.setActionN(u);
}

void WQ_Learning::endEpisode(const Reward& reward)
{
    double r = reward[0];
    double target = r;

    updateMeanAndSampleStdQ(target);
}

WQ_Learning::~WQ_Learning()
{
}

void WQ_Learning::init()
{
    FiniteTD::init();

    idxs = arma::mat(task.actionsNumber, task.actionsNumber - 1, arma::fill::zeros);
    arma::vec actions = arma::linspace(0, idxs.n_cols, idxs.n_rows);
    for(unsigned int i = 0; i < idxs.n_rows; i++)
        idxs.row(i) = actions(arma::find(actions != i)).t();

    sampleStdQ = Q + stdInfValue;
    Q2 = Q;
    weightsVar = Q;
    nUpdates = Q;
}

inline void WQ_Learning::updateMeanAndSampleStdQ(double target)
{
    double alpha = this->alpha(x, u);
    Q(x, u) = (1 - alpha) * Q(x, u) + alpha * target;
    Q2(x, u) = (1 - alpha) * Q2(x, u) + alpha * target * target;

    nUpdates(x, u)++;

    if(nUpdates(x, u) > 1)
    {
        weightsVar(x, u) = (1 - alpha) * (1 - alpha) * weightsVar(x, u) + alpha * alpha;
        double n = 1 / weightsVar(x, u);

        double var = (Q2(x, u) - Q(x, u) * Q(x, u)) / n;

        if(var >= stdZeroValue * stdZeroValue)
            sampleStdQ(x, u) = sqrt(var);
        else
            sampleStdQ(x, u) = stdZeroValue;
    }
}

}
