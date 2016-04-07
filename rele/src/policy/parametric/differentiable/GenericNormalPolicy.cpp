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


#include "rele/policy/parametric/differentiable/GenericNormalPolicy.h"
#include "rele/utils/RandomGenerator.h"

namespace ReLe
{

#define MVN_MEAN_GRAD_MACRO                                            \
    updateInternalState(state);                                        \
    arma::vec delta = action - mean;                                   \
    arma::mat gMu = meanApproximator.diff(state);                      \
    gMu.reshape(meanApproximator.getParametersSize(), mean.n_elem);    \
    arma::vec muGrad =  0.5 * gMu * (invSigma + invSigma.t()) * delta;

///////////////////////////////////////////////////////////////////////////////////////
/// MVN POLICY
///////////////////////////////////////////////////////////////////////////////////////

double GenericMVNPolicy::operator()(const arma::vec& state, const arma::vec& action)
{
    updateInternalState(state);
    return mvnpdfFast(action, mean, invSigma, determinant);
}

arma::vec GenericMVNPolicy::operator() (const arma::vec& state)
{
    updateInternalState(state);
    return mvnrandFast(mean, choleskySigma);
}

arma::vec GenericMVNPolicy::diff(const arma::vec &state, const arma::vec &action)
{
    return (*this)(state,action) * difflog(state,action);
}

arma::vec GenericMVNPolicy::difflog(const arma::vec &state, const arma::vec &action)
{
    MVN_MEAN_GRAD_MACRO;
    return muGrad;
}

arma::mat GenericMVNPolicy::diff2log(const arma::vec& state, const arma::vec& action)
{
    //TODO [IMPORTANT] IMPLEMENT!
    return arma::mat();
}

///////////////////////////////////////////////////////////////////////////////////////
/// Generic MVN POLICY with Diagonal covariance (parameters: mean, diagonal standard deviations)
///////////////////////////////////////////////////////////////////////////////////////

void GenericMVNDiagonalPolicy::setParameters(const arma::vec& w)
{
    assert(w.n_elem == this->getParametersSize());
    int dp = meanApproximator.getParametersSize();
    arma::vec tmp = w.rows(0, dp-1);
    meanApproximator.setParameters(tmp);
    for (int i = 0, ie = stddevParams.n_elem; i < ie; ++i)
    {
        stddevParams(i) = w[dp + i];
        assert(!std::isnan(stddevParams(i)) && !std::isinf(stddevParams(i)));
    }
    UpdateCovarianceMatrix();
}

arma::vec GenericMVNDiagonalPolicy::difflog(const arma::vec& state, const arma::vec& action)
{
    MVN_MEAN_GRAD_MACRO;

    arma::vec sigmaGrad(Sigma.n_rows);
    for (unsigned i = 0, ie = Sigma.n_rows; i < ie; ++i)
    {
        sigmaGrad[i] = - 1.0 / stddevParams(i) + (delta[i] * delta[i]) /  (stddevParams(i) * stddevParams(i) * stddevParams(i));
    }

    return vectorize(muGrad, sigmaGrad);
}

arma::mat GenericMVNDiagonalPolicy::diff2log(const arma::vec& state, const arma::vec& action)
{
    //TODO [IMPORTANT] IMPLEMENT!
    return arma::mat();
}

void GenericMVNDiagonalPolicy::UpdateCovarianceMatrix()
{
    determinant = 1.0;
    for (unsigned int i = 0, ie = stddevParams.n_elem; i < ie; ++i)
    {
        Sigma(i,i) = std::max(1e-10, stddevParams(i) * stddevParams(i));
        invSigma(i,i) = 1.0 / Sigma(i,i);
        // check that the covariance is not NaN or Inf
        assert(!std::isnan(Sigma(i,i)) && !std::isinf(Sigma(i,i)));
        assert(!std::isnan(invSigma(i,i)) && !std::isinf(invSigma(i,i)));
        determinant *= Sigma(i,i);
        choleskySigma(i,i) = sqrt(Sigma(i,i));
    }
//    mCholeskyDec = arma::chol(Sigma);
}

///////////////////////////////////////////////////////////////////////////////////////
/// Generic MVN POLICY with state dependant diagonal standard deviation (parameters: mean, std dev weights)
///////////////////////////////////////////////////////////////////////////////////////

arma::vec GenericMVNStateDependantStddevPolicy::difflog(const arma::vec& state, const arma::vec& action)
{

    MVN_MEAN_GRAD_MACRO;

    arma::mat gStd = stdApproximator.diff(state);
    gStd.reshape(stdApproximator.getParametersSize(), Sigma.n_rows);

    // compute gradient w.r.t. std weights
    arma::vec gNormStd(Sigma.n_rows);
    for (unsigned int i = 0, ie = Sigma.n_rows; i < ie; ++i)
    {
        double diagStdDev = sqrt(Sigma(i,i));
        gNormStd(i) = - 1.0 / diagStdDev + (delta[i] * delta[i]) /  (diagStdDev * Sigma(i,i));
    }
    arma::vec sigmaGrad = gStd*gNormStd;

    return vectorize(muGrad, sigmaGrad);
}

arma::mat GenericMVNStateDependantStddevPolicy::diff2log(const arma::vec &state, const arma::vec &action)
{
    //TODO [IMPORTANT] IMPLEMENT!
    return arma::mat();
}

void GenericMVNStateDependantStddevPolicy::updateInternalState(const arma::vec& state, bool cholesky_dec)
{
    // compute diagonal covariance matrix from std vector
    determinant = 1.0;
    arma::vec diagVec = stdApproximator(state);
    for (int i = 0, ie = diagVec.n_elem; i < ie; ++i)
    {
        Sigma(i,i) = std::max(1e-10, diagVec(i)*diagVec(i)); //diag covariance
        invSigma(i,i) = 1 / Sigma(i,i); //diag inverse
        // check that the covariance is not NaN or Inf
        assert(!std::isnan(Sigma(i,i)) && !std::isinf(Sigma(i,i)));
        assert(!std::isnan(invSigma(i,i)) && !std::isinf(invSigma(i,i)));
        determinant *= Sigma(i,i);
        choleskySigma(i,i) = sqrt(Sigma(i,i));
    }

    // compute mean vector
    mean = meanApproximator(state);
}

#undef MVN_MEAN_GRAD_MACRO

}
