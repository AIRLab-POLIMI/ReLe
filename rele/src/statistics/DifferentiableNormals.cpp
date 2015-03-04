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
    : DifferentiableDistribution(support_dim, param_size),
      parameters(param_size, fill::zeros),
      mean(support_dim, fill::zeros),
      Cov(support_dim, support_dim, fill::eye)
{
    UpdateInternalState();
}

ParametricNormal::ParametricNormal(vec& params, mat& covariance)
    : ParametricNormal(params.n_elem, params.n_elem)
{
    //    std::cout << "...." << params << std::endl;
    parameters = params;
    Cov        = covariance;
    invCov     = inv(Cov);
    detValue   = det(Cov);
    UpdateInternalState();
}

vec ParametricNormal::operator() ()
{
//    cerr << "Mean: " << mean;
//    cerr << "---------" << endl;
//    cerr << "Cov: " << Cov << endl;
    return mvnrand(mean, Cov);
}

double ParametricNormal::operator() (vec& point)
{
    return mvnpdfFast(point, mean, invCov, detValue);
}

void ParametricNormal::update(vec &increment)
{
    parameters += increment;
    this->UpdateInternalState();
}

vec ParametricNormal::difflog(const vec& point)
{
    return invCov * (point - parameters);
}

mat ParametricNormal::diff2Log(const vec&point)
{
    return -invCov;
}

void ParametricNormal::WriteOnStream(ostream& out)
{
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

void ParametricNormal::ReadFromStream(istream& in)
{
    double val;
    in >> pointSize;
    paramSize  = pointSize;
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

    UpdateInternalState();
}

void ParametricNormal::UpdateInternalState()
{
    mean = parameters;
}


////////////// PARAMETRIC LOGISTIC NORMAL DISTRIBUTION ///////////////////

ParametricLogisticNormal::ParametricLogisticNormal(unsigned int point_dim, double variance_asymptote)
    : ParametricNormal(point_dim, 2*point_dim),
      asVariance(variance_asymptote)
{
    mean = vec(point_dim,fill::zeros);
    UpdateInternalState();
}

ParametricLogisticNormal::ParametricLogisticNormal(unsigned int point_dim, double variance_asymptote, vec& params)
    : ParametricNormal(point_dim, 2*point_dim),
      asVariance(variance_asymptote)
{
    parameters = params;
    UpdateInternalState();
}

vec ParametricLogisticNormal::difflog(const vec& point)
{
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

void ParametricLogisticNormal::WriteOnStream(ostream &out)
{
    out << "ParametricLogisticNormal " << std::endl;
    out << pointSize << " " << paramSize << " " << asVariance << std::endl;
    for (unsigned i = 0; i < paramSize; ++i)
    {
        out << parameters(i) << " ";
    }
}

void ParametricLogisticNormal::ReadFromStream(istream &in)
{
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

    UpdateInternalState();
}

void ParametricLogisticNormal::UpdateInternalState()
{
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

} //end namespace
