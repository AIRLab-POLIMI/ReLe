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

#ifndef INCLUDE_RELE_IRL_UTILS_HESSIAN_HESSIANREINFORCE_H_
#define INCLUDE_RELE_IRL_UTILS_HESSIAN_HESSIANREINFORCE_H_

#include "rele/IRL/utils/hessian/step_based/StepBasedHessianCalculator.h"

namespace ReLe
{
template<class ActionC, class StateC>
class HessianReinforce : public StepBasedHessianCalculator<ActionC, StateC>
{
protected:
    USING_STEP_BASED_H_CALCULATORS_MEMBERS(ActionC, StateC)

public:
    HessianReinforce(Features& phi,
                     Dataset<ActionC,StateC>& data,
                     DifferentiablePolicy<ActionC,StateC>& policy,
                     double gamma) : StepBasedHessianCalculator<ActionC, StateC>(phi, data, policy, gamma)
    {

    }

    virtual ~HessianReinforce()
    {

    }

protected:
    virtual arma::cube computeHessianDiff() override
    {
        unsigned int dp = policy.getParametersSize();
        unsigned int dr = phi.rows();
        unsigned int episodeN = data.size();

        arma::cube Hdiff(dp, dp, phi.rows(), arma::fill::zeros);

        arma::mat Rew = data.computeEpisodeFeatureExpectation(phi, gamma);

        for(unsigned int ep = 0; ep < episodeN; ep++)
        {
            arma::mat G = this->computeG(data[ep]);

            for(unsigned int f = 0; f < phi.rows(); f++)
                Hdiff.slice(f) += G*Rew(f, ep);
        }

        Hdiff /= episodeN;

        return Hdiff;
    }



};

template<class ActionC, class StateC>
class HessianReinforceBase : public StepBasedHessianCalculator<ActionC, StateC>
{
protected:
    USING_STEP_BASED_H_CALCULATORS_MEMBERS(ActionC, StateC)

public:
    HessianReinforceBase(Features& phi,
                         Dataset<ActionC,StateC>& data,
                         DifferentiablePolicy<ActionC,StateC>& policy,
                         double gamma) : StepBasedHessianCalculator<ActionC, StateC>(phi, data, policy, gamma)
    {

    }

    virtual ~HessianReinforceBase()
    {

    }

protected:
    virtual arma::cube computeHessianDiff() override
    {
        unsigned int dp = policy.getParametersSize();
        unsigned int dr = phi.rows();
        unsigned int episodeN = data.size();

        arma::cube Hdiff(dp, dp, dr, arma::fill::zeros);
        arma::mat Rew = data.computeEpisodeFeatureExpectation(phi, gamma);

        arma::cube baseline_num(dp, dp, dr, arma::fill::zeros);
        arma::mat baseline_den(dp, dp, arma::fill::zeros);
        arma::cube G_ep(dp, dp, episodeN);

        for(unsigned int ep = 0; ep < episodeN; ep++)
        {
            // compute hessian essential
            arma::mat G = this->computeG(data[ep]);
            arma::mat G2 = G % G;

            // store hessian essentials
            G_ep.slice(ep) = G;
            baseline_den += G2;

            for(unsigned int r = 0; r < dr; r++)
                baseline_num.slice(r) += Rew(r, ep)*G2;
        }

        // compute the hessian
        arma::cube baseline = baseline_num.each_slice() / baseline_den;
        baseline(arma::find_nonfinite(baseline)).zeros();

        for (int ep = 0; ep < episodeN; ep++)
        {
            for(int r = 0; r < phi.rows(); r++)
                Hdiff.slice(r) += (Rew(r, ep) - baseline.slice(r)) % G_ep.slice(ep);
        }

        // compute mean values
        Hdiff /= episodeN;

        return Hdiff;
    }
};


template<class ActionC, class StateC>
class HessianReinforceTraceBaseSingle : public StepBasedHessianCalculator<ActionC, StateC>
{
protected:
    USING_STEP_BASED_H_CALCULATORS_MEMBERS(ActionC, StateC)

public:
    HessianReinforceTraceBaseSingle(Features& phi,
                                    Dataset<ActionC,StateC>& data,
                                    DifferentiablePolicy<ActionC,StateC>& policy,
                                    double gamma) : StepBasedHessianCalculator<ActionC, StateC>(phi, data, policy, gamma)
    {

    }

    virtual ~HessianReinforceTraceBaseSingle()
    {

    }

protected:
    virtual arma::cube computeHessianDiff() override
    {
        unsigned int dp = policy.getParametersSize();
        unsigned int dr = phi.rows();
        unsigned int episodeN = data.size();

        arma::cube Hdiff(dp, dp, dr, arma::fill::zeros);
        arma::mat Rew = data.computeEpisodeFeatureExpectation(phi, gamma);

        arma::vec baseline_num(dr, arma::fill::zeros);
        double baseline_den = 0;
        arma::cube G_ep(dp, dp, episodeN);

        for(unsigned int ep = 0; ep < episodeN; ep++)
        {
            // compute hessian essential
            arma::mat G = this->computeG(data[ep]);
            double trG = arma::trace(G);
            double trG2 = trG*trG;

            // store hessian essentials
            G_ep.slice(ep) = G;

            baseline_den += trG2;

            for(unsigned int r = 0; r < dr; r++)
                baseline_num(r) += Rew(r, ep)*trG2;
        }

        // compute the hessian
        arma::vec baseline = baseline_num / baseline_den;
        baseline(arma::find_nonfinite(baseline)).zeros();

        for (int ep = 0; ep < episodeN; ep++)
        {
            for(int r = 0; r < phi.rows(); r++)
                Hdiff.slice(r) += (Rew(r, ep) - baseline(r)) * G_ep.slice(ep);
        }

        // compute mean values
        Hdiff /= episodeN;

        return Hdiff;
    }
};

template<class ActionC, class StateC>
class HessianReinforceTraceBaseDiag : public StepBasedHessianCalculator<ActionC, StateC>
{
protected:
    USING_STEP_BASED_H_CALCULATORS_MEMBERS(ActionC, StateC)

public:
    HessianReinforceTraceBaseDiag(Features& phi,
                                  Dataset<ActionC,StateC>& data,
                                  DifferentiablePolicy<ActionC,StateC>& policy,
                                  double gamma) : StepBasedHessianCalculator<ActionC, StateC>(phi, data, policy, gamma)
    {

    }

    virtual ~HessianReinforceTraceBaseDiag()
    {

    }

protected:
    virtual arma::cube computeHessianDiff() override
    {
        unsigned int dp = policy.getParametersSize();
        unsigned int dr = phi.rows();
        unsigned int episodeN = data.size();

        arma::cube Hdiff(dp, dp, dr, arma::fill::zeros);
        arma::mat Rew = data.computeEpisodeFeatureExpectation(phi, gamma);

        arma::mat baseline_num(dp, dr, arma::fill::zeros);
        arma::mat baseline_den(dp, dp, arma::fill::zeros);
        arma::cube G_ep(dp, dp, episodeN);

        for(unsigned int ep = 0; ep < episodeN; ep++)
        {
            // compute hessian essential
            arma::mat G = this->computeG(data[ep]);
            arma::vec g = G.diag();
            double trG = arma::trace(G);

            // store hessian essentials
            G_ep.slice(ep) = G;

            baseline_den += g*g.t();

            for(unsigned int r = 0; r < dr; r++)
                baseline_num.col(r) += Rew(r, ep)*trG*g;
        }

        //normalize baseline components for numerical stability
        baseline_num /= episodeN;
        baseline_den /= episodeN;

        // compute the hessian
        arma::mat baseline;

        if(arma::rank(baseline_den) == dp)
            baseline = arma::solve(baseline_den, baseline_num);
        else
            baseline = arma::pinv(baseline_den)*baseline_num;

        for (int ep = 0; ep < episodeN; ep++)
        {
            for(int r = 0; r < phi.rows(); r++)
                Hdiff.slice(r) += (Rew(r, ep) - arma::diagmat(baseline.col(r))) % G_ep.slice(ep);
        }

        // compute mean values
        Hdiff /= episodeN;

        return Hdiff;
    }
};


}



#endif /* INCLUDE_RELE_IRL_UTILS_HESSIAN_HESSIANREINFORCE_H_ */
