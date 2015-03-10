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

#include "DifferentiableNormals.h"
#include "ArmadilloPDFs.h"

using namespace std;
using namespace arma;

namespace ReLe
{

////////////// PARAMETRIC NORMAL DISTRIBUTION ///////////////////

ParametricNormal::ParametricNormal(unsigned int support_dim, unsigned int param_size)
    : DifferentiableDistribution(support_dim),
      parameters(param_size, fill::zeros),
      mean(support_dim, fill::zeros),
      Cov(support_dim, support_dim, fill::eye),
      cholCov(arma::chol(Cov))
{
    updateInternalState();
}

ParametricNormal::ParametricNormal(vec& params, mat& covariance)
    : ParametricNormal(params.n_elem, params.n_elem)
{
    //    std::cout << "...." << params << std::endl;
    parameters = params;
    Cov        = covariance;
    invCov     = inv(Cov);
    detValue   = det(Cov);
    updateInternalState();
}

vec ParametricNormal::operator() ()
{
    //    cerr << "Mean: " << mean;
    //    cerr << "---------" << endl;
    //    cerr << "Cov: " << Cov << endl;
    return mvnrandFast(mean, cholCov);
}

double ParametricNormal::operator() (vec& point)
{
    return mvnpdfFast(point, mean, invCov, detValue);
}

void ParametricNormal::update(vec &increment)
{
    parameters += increment;
    this->updateInternalState();
}

vec ParametricNormal::difflog(const vec& point)
{
    return invCov * (point - parameters);
}

mat ParametricNormal::diff2Log(const vec&point)
{
    return -invCov;
}

void ParametricNormal::writeOnStream(ostream& out)
{
    int paramSize = this->getParametersSize();
    out << "ParametricNormal " << std::endl;
    out << pointSize << std::endl;
    for (unsigned i = 0; i < paramSize; ++i)
    {
        out << parameters(i) << " ";
    }
    out << std::endl;
    for (unsigned i = 0; i < paramSize; ++i)
    {
        for (unsigned j = 0; j < paramSize; ++j)
        {
            out << Cov(i,j) << " ";
        }
    }
}

void ParametricNormal::readFromStream(istream& in)
{
    double val;
    in >> pointSize;
    int paramSize  = pointSize;
    parameters = zeros<vec>(paramSize);
    for (unsigned i = 0; i < paramSize; ++i)
    {
        in >> val;
        parameters(i) = val;
    }
    Cov = zeros<mat>(paramSize, paramSize);
    for (unsigned i = 0; i < paramSize; ++i)
    {
        for (unsigned j = 0; j < paramSize; ++j)
        {
            in >> val;
            Cov(i,j) = val;
        }
    }
    invCov = inv(Cov);
    detValue = det(Cov);

    updateInternalState();
}

void ParametricNormal::updateInternalState()
{
    mean = parameters;
}


////////////// PARAMETRIC LOGISTIC NORMAL DISTRIBUTION ///////////////////

ParametricLogisticNormal::ParametricLogisticNormal(unsigned int point_dim, double variance_asymptote)
    : ParametricNormal(point_dim, 2*point_dim),
      asVariance(variance_asymptote)
{
    mean = vec(point_dim,fill::zeros);
    updateInternalState();
}

ParametricLogisticNormal::ParametricLogisticNormal(unsigned int point_dim, double variance_asymptote, vec& params)
    : ParametricNormal(point_dim, 2*point_dim),
      asVariance(variance_asymptote)
{
    parameters = params;
    updateInternalState();
}

vec ParametricLogisticNormal::difflog(const vec& point)
{
    int paramSize = this->getParametersSize();
    vec diff(pointSize);
    for (unsigned int i = 0; i < pointSize; ++i)
    {
        diff[i] = point[i] - parameters[i];
    }
    vec mean_grad = invCov * diff;
    vec gradient(paramSize);

    for (unsigned int i = 0; i < pointSize; ++i)
    {
        gradient[i] = mean_grad(i);
        //        std::cout << "p(" << i << "): " << mParameters(i) << std::endl;
    }
    for (unsigned int i = pointSize, ie = paramSize; i < ie; ++i)
    {
        int idx    = i - pointSize;
        double val = parameters[i];
        //        std::cout << idx << std::endl;
        //        std::cout << "p(" << i << "): " << val << std::endl;
        //        double logisticVal = logistic(val, mAsVariance);
        //        std::cout << logisticVal << std::endl;
        double A = - 0.5 * exp(-val) / (1 + exp(-val));
        double B = 0.5 * exp(-val) * diff[idx] * diff[idx] / asVariance;
        gradient[i] = A + B;
    }
    return gradient;
}

mat ParametricLogisticNormal::diff2Log(const vec& point)
{
    int paramSize = this->getParametersSize();
    mat hessian(paramSize,paramSize,fill::zeros);
    for (unsigned i = 0; i < pointSize; ++i)
    {
        for (unsigned j = 0; j < pointSize; ++j)
        {
            hessian(i,j) = -invCov(i,j);
        }


        int idx = pointSize + i;
        double val = parameters(idx);
        double diff = point[i] - parameters[i];

        // consider only the components different from zero
        // obtained from the derivative of the gradient w.r.t. the covariance parameters
        // these terms are only mSupportSize
        // d2 logp / dpdp
        double A = 0.5 * exp(-val) / ((1+exp(-val))*(1+exp(-val)));
        double B = - 0.5 * exp(-val) * diff * diff / asVariance;
        hessian(idx,idx) = A + B;

        // d2 logp / dpdm
        hessian(i, pointSize+i) = - diff * exp(-val) / asVariance;

        // d2 logp / dmdp
        hessian(pointSize+i, i) = hessian(i, pointSize+i);
    }
}

void ParametricLogisticNormal::writeOnStream(ostream &out)
{
    int paramSize = this->getParametersSize();
    out << "ParametricLogisticNormal " << std::endl;
    out << pointSize << " " << paramSize << " " << asVariance << std::endl;
    for (unsigned i = 0; i < paramSize; ++i)
    {
        out << parameters(i) << " ";
    }
}

void ParametricLogisticNormal::readFromStream(istream &in)
{
    int paramSize;
    double val;
    in >> pointSize;
    in >> paramSize;
    in >> asVariance;
    parameters = vec(paramSize);
    for (unsigned i = 0; i < paramSize; ++i)
    {
        in >> val;
        parameters(i) = val;
    }

    updateInternalState();
}

void ParametricLogisticNormal::updateInternalState()
{
    int paramSize = this->getParametersSize();
    cerr << "asVariance: " << asVariance << endl;
    //    Cov.zeros();
    for (unsigned i = 0; i < pointSize; ++i)
    {
        mean(i) = parameters(i);
    }
    for (int i = pointSize, ie = paramSize; i < ie; ++i)
    {
        int idx = i - pointSize;
        Cov(idx,idx) = logistic(parameters(i), asVariance);
    }
    invCov = arma::inv(Cov);
    detValue = arma::det(Cov);
}


////////////// PARAMETRIC CHOLESKY NORMAL DISTRIBUTION ///////////////////

ParametricCholeskyNormal::ParametricCholeskyNormal(unsigned int point_dim, arma::vec& initial_mean, arma::mat& initial_cholA)
    :ParametricNormal(point_dim, 2*point_dim + (point_dim * point_dim - point_dim) / 2)
{
    arma::mat tmp = arma::trimatu(arma::ones(point_dim, point_dim));
    arma::vec val = initial_cholA.elem( find(tmp == 1.0) );
    int dp = 2*point_dim + (point_dim * point_dim - point_dim) / 2;
    parameters = arma::vec(dp);
    for (int i = 0; i < point_dim; ++i)
    {
        parameters[i] = initial_mean[i];
    }
    for (int i = point_dim; i < dp; ++i)
    {
        parameters[i] = val[i-point_dim];
    }

    cout << parameters << endl;
    this->updateInternalState();
}

arma::vec ParametricCholeskyNormal::difflog(const arma::vec &point)
{
    int paramSize = this->getParametersSize();
    vec gradient(paramSize);

    //--- mean gradient
    vec diff = point - mean;
    vec mean_grad = invCov * diff;
    //---

    //--- Covariance gradient
    arma::mat tmp = (point - mean) * (point - mean).t() * invCov;
    arma::mat R = arma::solve(cholCov.t(), tmp);
    arma::mat dlogpdt_sigma(pointSize, pointSize, arma::fill::zeros);
    for (int i = 0; i < pointSize; ++i)
        for (int j = 0; j < pointSize; ++j)
            if (i == j)
                dlogpdt_sigma(i,j) = R(i,j) - 1.0 / cholCov(i,j);
            else
                dlogpdt_sigma(i,j) = R(i,j);
    arma::mat idxs = arma::trimatu(arma::ones(pointSize, pointSize));
    arma::vec vals = dlogpdt_sigma.elem( find(idxs == 1) );
    //---


    for (unsigned int i = 0; i < pointSize; ++i)
    {
        gradient[i] = mean_grad(i);
    }
    for (int i = pointSize; i < paramSize; ++i)
    {
        gradient[i] = vals[i-pointSize];
    }
}

arma::mat ParametricCholeskyNormal::diff2Log(const arma::vec &point)
{

}

arma::sp_mat ParametricCholeskyNormal::FIM()
{
    //TODO: make in a more efficient way
    int rows = invCov.n_rows;
    int cols = invCov.n_cols;
    vector<mat> diag_blocks;
    diag_blocks.push_back(invCov);
    for (int k = 0; k < pointSize; ++k)
    {
        int index = pointSize - k;
        int low_index = pointSize - index;
        arma::mat tmp = invCov( span(low_index, pointSize-1), span(low_index, pointSize-1) );
        tmp(0,0) += 1.0 / (cholCov(k,k)*cholCov(k,k));
        rows += tmp.n_rows;
        cols += tmp.n_cols;
        diag_blocks.push_back(tmp);
    }
    arma::sp_mat fim(rows, cols);
    int roffset = 0, coffset = 0;
    for (int i = 0, ie = diag_blocks.size(); i < ie; ++i)
    {
        arma::mat& mtx = diag_blocks[i];
        for (int r = 0, re = mtx.n_rows; r < re; ++r)
        {
            for (int c = 0, ce = mtx.n_cols; c < ce; ++c)
            {
                fim(roffset+r, coffset+c) = mtx(r,c);
            }
        }
        roffset = roffset + mtx.n_rows;
        coffset = coffset + mtx.n_cols;
    }
    return fim;
}

arma::sp_mat ParametricCholeskyNormal::inverseFIM()
{
    //TODO: make in a more efficient way
    int rows = Cov.n_rows;
    int cols = Cov.n_cols;
    vector<mat> diag_blocks;
    diag_blocks.push_back(Cov);
    for (int k = 0; k < pointSize; ++k)
    {
        int index = pointSize - k;
        int low_index = pointSize - index;
        arma::mat tmp = invCov( span(low_index, pointSize-1), span(low_index, pointSize-1) );
        tmp(0,0) += 1.0 / (cholCov(k,k)*cholCov(k,k));
        arma::mat nMtx(tmp.n_rows, tmp.n_cols);
        for (int r = 0; r < tmp.n_rows; ++r)
        {
            for (int c = 0; c < tmp.n_cols; ++c)
            {
                if (tmp(r,c) != 0.0)
                    nMtx(r,c) = 1 / tmp(r,c);
                else
                    nMtx(r,c) = 0.0;
            }
        }
        rows += nMtx.n_rows;
        cols += nMtx.n_cols;
        diag_blocks.push_back(nMtx);
    }
    arma::sp_mat fim(rows, cols);
    int roffset = 0, coffset = 0;
    for (int i = 0, ie = diag_blocks.size(); i < ie; ++i)
    {
        arma::mat& mtx = diag_blocks[i];
        for (int r = 0, re = mtx.n_rows; r < re; ++r)
        {
            for (int c = 0, ce = mtx.n_cols; c < ce; ++c)
            {
                fim(roffset+r, coffset+c) = mtx(r,c);
            }
        }
        roffset = roffset + mtx.n_rows;
        coffset = coffset + mtx.n_cols;
    }
    return fim;
}

void ParametricCholeskyNormal::writeOnStream(ostream &out)
{
    int paramSize = this->getParametersSize();
    out << "ParametricCholeskyNormal" << std::endl;
    out << pointSize << std::endl;
    out << paramSize << std::endl;
    for (unsigned i = 0; i < paramSize; ++i)
    {
        out << parameters(i) << "\t";
    }
    out << std::endl;
}

void ParametricCholeskyNormal::readFromStream(istream &in)
{
    double val;
    in >> pointSize;
    int paramSize;
    in >> paramSize;
    parameters = zeros<vec>(paramSize);
    for (unsigned i = 0; i < paramSize; ++i)
    {
        in >> val;
        parameters[i] = val;
    }
}

void ParametricCholeskyNormal::updateInternalState()
{
    int paramSize = this->getParametersSize();
    for (unsigned i = 0; i < pointSize; ++i)
    {
        mean(i) = parameters(i);
    }
    //TODO: fare in modo piu' efficiente
    arma::mat tmp = arma::trimatu(arma::ones(pointSize, pointSize));
    cholCov.elem( find(tmp == 1.0) ) = parameters.rows(pointSize, paramSize-1);
    Cov = cholCov.t() * cholCov;

    //TODO: questo si potrebbe fare meglio
    invCov = arma::inv(Cov);
    detValue = arma::det(Cov);
}

} //end namespace
