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

/*
 * Written by: Carlo D'Eramo
 */

#include "td/WQ-Learning.h"

using namespace std;
using namespace arma;


struct pars
{
    unsigned int xn;
    unsigned int u;
    arma::mat idxs;
    arma::mat meanQ;
    arma::mat sampleStdQ;
};

double fun(double x, void* params)
{
    pars p = *(pars*) params;
    const unsigned int xn = p.xn;
    const unsigned int u = p.u;
    const arma::mat& idxs = p.idxs;
    const arma::mat& meanQ = p.meanQ;
    const arma::mat& sampleStdQ = p.sampleStdQ;

    arma::vec cdf(idxs.n_cols, arma::fill::zeros);
    for(unsigned int i = 0; i < cdf.n_elem; i++)
        cdf(i) = gsl_cdf_gaussian_P(x - meanQ(xn, idxs(u, i)), sampleStdQ(xn, idxs(u, i)));
    double f = gsl_ran_gaussian_pdf(x - meanQ(xn, u), sampleStdQ(xn, u)) * arma::prod(cdf);

    return f;
}


namespace ReLe
{

WQ_Learning::WQ_Learning(ActionValuePolicy<FiniteState>& policy) :
    Q_Learning(policy)
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

    gsl_integration_workspace* w = gsl_integration_workspace_alloc(1000);
    double delta;
    if(nUpdatesQ(x, u) > 0)
    {
        arma::vec integrals(task.finiteActionDim, arma::fill::zeros);
        pars p;
        p.xn = (unsigned int) xn;
        p.meanQ = meanQ;
        p.sampleStdQ = sampleStdQ;
        p.idxs = idxs;
        for(unsigned int i = 0; i < integrals.n_elem; i++)
        {
            gsl_function f;
            f.function = &fun;
            p.u = i;
            f.params = &p;

            double result, error;
            double lowerLimit = meanQ(xn, i) - 2 * sampleStdQ(xn, i);
            double upperLimit = meanQ(xn, i) + 2 * sampleStdQ(xn, i);
            gsl_integration_qags(&f, lowerLimit, upperLimit, 0, 1e-8, 1000, w, &result, &error);

            integrals(i) = result;
        }
        double W = arma::dot(integrals, Q.row(xn));

        delta = r + task.gamma * W - Q(x, u);
    }
    else
        delta = r - Q(x, u);

    gsl_integration_workspace_free(w);

    Q(x, u) = Q(x, u) + alpha * delta;

    updateMeanAndSampleStdQ(Q(x, u));

    //update action and state
    x = xn;
    u = policy(xn);

    //set next action
    action.setActionN(u);
}

void WQ_Learning::endEpisode(const Reward& reward)
{
    //Last update
    double r = reward[0];
    double delta = r - Q(x, u);
    Q(x, u) = Q(x, u) + alpha * delta;

    updateMeanAndSampleStdQ(Q(x, u));
}

WQ_Learning::~WQ_Learning()
{
}

void WQ_Learning::init()
{
    FiniteTD::init();

    idxs = arma::mat(task.finiteActionDim, task.finiteActionDim - 1, arma::fill::zeros);
    arma::vec actions = arma::linspace(0, idxs.n_cols, idxs.n_rows);
    for(unsigned int i = 0; i < idxs.n_rows; i++)
        idxs.row(i) = actions(arma::find(actions != i)).t();

    meanQ = arma::mat(Q.n_rows, Q.n_cols, arma::fill::zeros);
    // There are 0 elements at the beginning, variance is infinite
    sampleStdQ = arma::mat(Q.n_rows, Q.n_cols).fill(STD_INF_VALUE);
    sumQ = arma::mat(Q.n_rows, Q.n_cols, arma::fill::zeros);
    sumSquareQ = arma::mat(Q.n_rows, Q.n_cols, arma::fill::zeros);
    nUpdatesQ = arma::mat(Q.n_rows, Q.n_cols, arma::fill::zeros);
}

inline void WQ_Learning::updateMeanAndSampleStdQ(double q)
{
    nUpdatesQ(x, u)++;

    if(nUpdatesQ(x, u) > 1)
    {
        sumQ(x, u) += q;
        sumSquareQ(x, u) += pow(q, 2);
        meanQ(x, u) = sumQ(x, u) / nUpdatesQ(x, u);
        double diff = sumSquareQ(x, u) - pow(sumQ(x, u), 2) / nUpdatesQ(x, u);

        if(diff > 0)
            sampleStdQ(x, u) = sqrt((diff / (nUpdatesQ(x, u) - 1)) / nUpdatesQ(x, u));
        else
            sampleStdQ(x, u) = STD_ZERO_VALUE;
    }
}

}
