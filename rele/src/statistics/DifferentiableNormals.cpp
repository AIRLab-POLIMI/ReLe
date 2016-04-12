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

#include "rele/statistics/DifferentiableNormals.h"
#include "rele/utils/ArmadilloPDFs.h"
#include "rele/utils/ArmadilloExtensions.h"
#include <cassert>

using namespace std;
using namespace arma;

namespace ReLe
{

///////////////////////////////////////////////////////////////////////////////////////
/// PARAMETRIC NORMAL DISTRIBUTION
///////////////////////////////////////////////////////////////////////////////////////

ParametricNormal::ParametricNormal(unsigned int support_dim)
    : DifferentiableDistribution(support_dim),
      mean(support_dim, fill::zeros),
      Cov(support_dim, support_dim, fill::eye),
      invCov(support_dim, support_dim, fill::eye),
      cholCov(support_dim, support_dim, fill::eye),
      detValue(0)
{

}

ParametricNormal::ParametricNormal(const vec& params, const mat& covariance)
    : ParametricNormal(params.n_elem)
{
    mean       = params;
    Cov        = covariance;
    invCov     = inv(Cov);
    detValue   = det(Cov);
    cholCov    = chol(Cov);
}

vec ParametricNormal::operator() () const
{
    //cerr << "Mean: " << mean;
    //cerr << "---------" << endl;
    //cerr << "Cov: " << Cov << endl;
    return mvnrandFast(mean, cholCov);
}

double ParametricNormal::operator() (const vec& point) const
{
    return mvnpdfFast(point, mean, invCov, detValue);
}

void ParametricNormal::wmle(const arma::vec& weights, const arma::mat& samples)
{
    mean = samples*weights/sum(weights);
    updateInternalState();
}

void ParametricNormal::setParameters(const arma::vec& newval)
{
    mean = newval;
}

void ParametricNormal::update(const vec &increment)
{
    mean += increment;
    this->updateInternalState();
}

vec ParametricNormal::difflog(const vec& point) const
{
    return invCov * (point - mean);
}

mat ParametricNormal::diff2log(const vec&point) const
{
    return -invCov;
}

vec ParametricNormal::pointDifflog(const vec& point) const
{
    return invCov * (point - mean);
}

void ParametricNormal::writeOnStream(ostream& out)
{
    out << "ParametricNormal " << std::endl;
    out << pointSize << std::endl;
    for (unsigned i = 0; i < pointSize; ++i)
    {
        out << mean(i) << " ";
    }
    out << std::endl;
    for (unsigned i = 0; i < pointSize; ++i)
    {
        for (unsigned j = 0; j < pointSize; ++j)
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
    mean = zeros<vec>(paramSize);
    for (unsigned i = 0; i < paramSize; ++i)
    {
        in >> val;
        mean(i) = val;
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
}

void ParametricNormal::updateInternalState()
{
}

///////////////////////////////////////////////////////////////////////////////////////
/// DIAGONAL COVARIANCE NORMAL DISTRIBUTION
///////////////////////////////////////////////////////////////////////////////////////
ParametricDiagonalNormal::ParametricDiagonalNormal(const arma::vec& mean, const arma::vec& standardeviation)
    : ParametricNormal(mean.n_elem), diagStdDev(standardeviation)
{
    assert(mean.n_elem == standardeviation.n_elem);
    this->mean = mean;
    updateInternalState();
}

void ParametricDiagonalNormal::wmle(const arma::vec& weights, const arma::mat& samples)
{
    double sumD = sum(weights);
    double sumD2 = sum(square(weights));
    double Z = sumD - sumD2/sumD;

    mean = samples*weights/sumD;

    arma::mat delta = samples;
    delta.each_col() -= mean;
    diagStdDev = square(delta)*weights/Z;

    updateInternalState();

}

arma::vec ParametricDiagonalNormal::difflog(const arma::vec& point) const
{
    vec gradient(pointSize*2);

    for (unsigned int i = 0; i < pointSize; ++i)
    {
        gradient[i] = (point[i] - mean[i])/(diagStdDev(i) * diagStdDev(i));
        //        std::cout << "p(" << i << "): " << mParameters(i) << std::endl;
    }
    for (unsigned int i = 0, ie = pointSize; i < ie; ++i)
    {
        double val = point[i] - mean[i];
        //        std::cout << idx << std::endl;
        gradient[i+pointSize] = - 1.0 / diagStdDev(i) + (val * val) /  (diagStdDev(i) * diagStdDev(i) * diagStdDev(i));
    }
    return gradient;
}

arma::mat ParametricDiagonalNormal::diff2log(const arma::vec& point) const
{
    //TODO [IMPORTANT] controllare implementazione
    int paramSize = this->getParametersSize();
    mat hessian(paramSize,paramSize,fill::zeros);

    for (unsigned i = 0; i < pointSize; ++i)
    {
        // d2 logp / dmdm
        hessian(i,i) = -invCov(i,i);

        int idx = pointSize + i;
        double diff = point[i] - mean[i];

        // consider only the components different from zero
        // obtained from the derivative of the gradient w.r.t. the covariance parameters
        // these terms are only mSupportSize
        // d2 logp / dpdp
        double sigma2 = diagStdDev(i) * diagStdDev(i);
        hessian(idx,idx) = 1.0 / (sigma2) - 3.0 * (diff * diff) /  (sigma2 * sigma2);;

        // d2 logp / dpdm
        hessian(i, pointSize+i) = - 2.0 * diff / (sigma2 * diagStdDev(i));

        // d2 logp / dmdp
        hessian(pointSize+i, i) = hessian(i, pointSize+i);
    }
    return hessian;
}

sp_mat ParametricDiagonalNormal::FIM() const
{
    //TODO [OPTIMIZATION] make in a more efficient way
    int rows = invCov.n_rows;
    int cols = invCov.n_cols;
    vector<mat> diag_blocks;
    diag_blocks.push_back(invCov);
    for (int k = 0, end_v = pointSize - 1; k < pointSize; ++k)
    {
        mat tmp = invCov( span(k, end_v), span(k, end_v) );
        tmp(0,0) += 1.0 / (cholCov(k,k)*cholCov(k,k));
        rows += tmp.n_rows;
        cols += tmp.n_cols;
        diag_blocks.push_back(tmp);
    }
    //==========================
    sp_mat tmp = blockdiagonal(diag_blocks, rows, cols);
    //=========================

    //Construct list of indexes that must be kept
    uvec indexes(2*pointSize);
    for (int i = 0; i < pointSize; ++i)
        indexes(i) = i;
    indexes(pointSize) = pointSize;
    for (int i = 1; i < pointSize; ++i)
    {
        indexes(pointSize+i) = indexes(pointSize+i-1) + pointSize - i + 1;
    }

    //Select subview of the sparse matrix
    sp_mat final(2*pointSize,2*pointSize);
    int r, c;
    for (int i = 0, dim = 2*pointSize; i < dim; ++i)
    {
        r = indexes(i);
        for (int j = 0; j < dim; ++j)
        {
            c = indexes(j);
            final(i,j) = tmp.at(r,c);
        }
    }

    return final;
}

sp_mat ParametricDiagonalNormal::inverseFIM() const
{
    //TODO [OPTIMIZATION] make in a more efficient way
    int rows = Cov.n_rows;
    int cols = Cov.n_cols;
    vector<mat> diag_blocks;
    diag_blocks.push_back(Cov);
    for (int k = 0, end_v = pointSize - 1; k < pointSize; ++k)
    {
        mat tmp = invCov( span(k, end_v), span(k, end_v) );
        tmp(0,0) += 1.0 / (cholCov(k,k)*cholCov(k,k));
        mat nMtx = arma::solve(tmp.t(), arma::eye(tmp.n_rows, tmp.n_cols)).t(); //in matlab it is mrdivide operator
        //std::cout << nMtx << std::endl << std::endl;
        rows += nMtx.n_rows;
        cols += nMtx.n_cols;
        diag_blocks.push_back(nMtx);
    }
    //==========================
    sp_mat tmp = blockdiagonal(diag_blocks, rows, cols);
    //=========================

    //Construct list of indexes that must be kept
    uvec indexes(2*pointSize);
    for (int i = 0; i < pointSize; ++i)
        indexes(i) = i;
    indexes(pointSize) = pointSize;
    for (int i = 1; i < pointSize; ++i)
    {
        indexes(pointSize+i) = indexes(pointSize+i-1) + pointSize - i + 1;
    }

    //Select subview of the sparse matrix
    sp_mat final(2*pointSize,2*pointSize);
    int r, c;
    for (int i = 0, dim = 2*pointSize; i < dim; ++i)
    {
        r = indexes(i);
        for (int j = 0; j < dim; ++j)
        {
            c = indexes(j);
            final(i,j) = tmp.at(r,c);
        }
    }

    return final;
}

void ParametricDiagonalNormal::writeOnStream(ostream& out)
{
    out << "ParametricDiagonalNormal " << std::endl;
    out << pointSize << std::endl;
    for (unsigned i = 0; i < pointSize; ++i)
    {
        out << mean(i) << " ";
    }
    for (unsigned i = 0; i < pointSize; ++i)
    {
        out << diagStdDev(i) << " ";
    }
    out << std::endl;
}

void ParametricDiagonalNormal::readFromStream(istream& in)
{
    double val;
    in >> pointSize;

    // allocate space
    mean.set_size(pointSize);
    diagStdDev.set_size(pointSize);
    Cov.eye(pointSize,pointSize);
    invCov.eye(pointSize,pointSize);
    cholCov.eye(pointSize,pointSize);

    for (unsigned i = 0; i < pointSize; ++i)
    {
        in >> val;
        mean(i) = val;
    }
    for (unsigned i = 0; i < pointSize; ++i)
    {
        in >> val;
        diagStdDev(i) = val;
    }
    updateInternalState();
}

unsigned int ParametricDiagonalNormal::getParametersSize() const
{
    return 2*mean.n_elem;
}

arma::vec ParametricDiagonalNormal::getParameters() const
{
    return arma::join_vert(mean, diagStdDev);
}

void ParametricDiagonalNormal::setParameters(const arma::vec& newval)
{
    assert(newval.n_elem == 2*mean.n_elem);
    int i, nb = mean.n_elem;
    for (i = 0; i < nb; ++i)
    {
        mean[i] = newval[i];
    }
    for (i = 0; i < nb; ++i)
    {
        diagStdDev[i] = newval[i+nb];
    }
    updateInternalState();
}

void ParametricDiagonalNormal::update(const arma::vec& increment)
{
    assert(increment.n_elem == 2*mean.n_elem);
    int i, nb = mean.n_elem;
    for (i = 0; i < nb; ++i)
    {
        mean[i] += increment[i];
    }
    for (i = 0; i < nb; ++i)
    {
        diagStdDev[i] += increment[i+nb];
    }
    updateInternalState();
}

void ParametricDiagonalNormal::updateInternalState()
{
    for (int i = 0, ie = pointSize; i < ie; ++i)
    {
        Cov(i,i) = diagStdDev(i)*diagStdDev(i);
        invCov(i,i) = 1.0/Cov(i,i);
        cholCov(i,i) = diagStdDev(i);
    }
}


///////////////////////////////////////////////////////////////////////////////////////
/// PARAMETRIC LOGISTIC NORMAL DISTRIBUTION
///////////////////////////////////////////////////////////////////////////////////////

ParametricLogisticNormal::ParametricLogisticNormal(unsigned int point_dim, double variance_asymptote)
    : ParametricNormal(point_dim),
      asVariance(variance_asymptote*ones<vec>(point_dim)), logisticWeights(point_dim, fill::zeros)
{
    updateInternalState();
}

ParametricLogisticNormal::ParametricLogisticNormal(const arma::vec& mean, const arma::vec& logWeights, double variance_asymptote)
    : ParametricNormal(mean.n_elem),
      asVariance(variance_asymptote*ones<vec>(mean.n_elem)), logisticWeights(logWeights)
{
    assert(mean.n_elem == logWeights.n_elem);
    this->mean = mean;
    updateInternalState();
}

ParametricLogisticNormal::ParametricLogisticNormal(const arma::vec& variance_asymptote)
    : ParametricNormal(variance_asymptote.n_elem),
      asVariance(variance_asymptote), logisticWeights(variance_asymptote.n_elem, fill::zeros)
{
    updateInternalState();
}

ParametricLogisticNormal::ParametricLogisticNormal(const arma::vec& mean, const arma::vec& logWeights, const arma::vec& variance_asymptote)
    : ParametricNormal(mean.n_elem),
      asVariance(variance_asymptote), logisticWeights(logWeights)
{
    assert(mean.n_elem == logWeights.n_elem);
    assert(mean.n_elem == variance_asymptote.n_elem);
    this->mean = mean;
    updateInternalState();
}

void ParametricLogisticNormal::wmle(const arma::vec& weights, const arma::mat& samples)
{
    //TODO [IMPORTANT] implement
}

vec ParametricLogisticNormal::difflog(const vec& point) const
{
    vec diff = point - mean;
    vec mean_grad = invCov * diff;
    vec gradient(2*pointSize);

    for (unsigned int i = 0, ie = pointSize; i < ie; ++i)
    {
        gradient[i] = mean_grad(i);
        double val = logisticWeights[i];
        double A = - 0.5 * exp(-val) / (1 + exp(-val));
        double B = 0.5 * exp(-val) * diff[i] * diff[i] / asVariance[i];
        gradient[i+pointSize] = A + B;
    }
    return gradient;
}

mat ParametricLogisticNormal::diff2log(const vec& point) const
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
        double val = logisticWeights(i);
        double diff = point[i] - mean[i];

        // consider only the components different from zero
        // obtained from the derivative of the gradient w.r.t. the covariance parameters
        // these terms are only mSupportSize
        // d2 logp / dpdp
        double A = 0.5 * exp(-val) / ((1+exp(-val))*(1+exp(-val)));
        double B = - 0.5 * exp(-val) * diff * diff / asVariance[i];
        hessian(idx,idx) = A + B;

        // d2 logp / dpdm
        hessian(i, pointSize+i) = - diff * exp(-val) / asVariance[i];

        // d2 logp / dmdp
        hessian(pointSize+i, i) = hessian(i, pointSize+i);
    }
    return hessian;
}

void ParametricLogisticNormal::writeOnStream(ostream &out)
{
    out << "ParametricLogisticNormal " << std::endl;
    out << pointSize << std::endl;
    for (unsigned i = 0; i < pointSize; ++i)
    {
        out << asVariance[i] << "\t";
    }
    out << std::endl;
    for (unsigned i = 0; i < pointSize; ++i)
    {
        out << mean(i) << " ";
    }
    for (unsigned i = 0; i < pointSize; ++i)
    {
        out << logisticWeights(i) << " ";
    }
}

void ParametricLogisticNormal::readFromStream(istream &in)
{
    double val;
    in >> pointSize;

    // allocate space
    asVariance.set_size(pointSize);
    mean.set_size(pointSize);
    logisticWeights.set_size(pointSize);
    Cov.eye(pointSize,pointSize);
    invCov.eye(pointSize,pointSize);
    cholCov.eye(pointSize,pointSize);


    for (unsigned i = 0; i < pointSize; ++i)
    {
        in >> asVariance[i];
    }
    for (unsigned i = 0; i < pointSize; ++i)
    {
        in >> val;
        mean(i) = val;
    }

    for (unsigned i = 0; i < pointSize; ++i)
    {
        in >> val;
        logisticWeights(i) = val;
    }

    updateInternalState();
}

unsigned int ParametricLogisticNormal::getParametersSize() const
{
    return 2*mean.n_elem;
}

arma::vec ParametricLogisticNormal::getParameters() const
{
    return arma::join_vert(mean, logisticWeights);
}

void ParametricLogisticNormal::setParameters(const arma::vec& newval)
{
    assert(newval.n_elem == 2*mean.n_elem);
    int i, nb = mean.n_elem;
    for (i = 0; i < nb; ++i)
    {
        mean[i] = newval[i];
    }
    for (i = 0; i < nb; ++i)
    {
        logisticWeights[i] = newval[i+nb];
    }
    updateInternalState();
}

void ParametricLogisticNormal::update(const arma::vec& increment)
{
    assert(increment.n_elem == 2*mean.n_elem);
    int i, nb = mean.n_elem;
    for (i = 0; i < nb; ++i)
    {
        mean[i] += increment[i];
    }
    for (i = 0; i < nb; ++i)
    {
        logisticWeights[i] += increment[i+nb];
    }
    updateInternalState();
}

void ParametricLogisticNormal::updateInternalState()
{
    for (int i = 0, ie = pointSize; i < ie; ++i)
    {
        Cov(i,i) = std::max(1e-6, logistic(logisticWeights(i), asVariance[i])); //to avoid numerical problems
        invCov(i,i) = 1.0/Cov(i,i);
        cholCov(i,i) = sqrt(Cov(i,i));
    }
    detValue = det(Cov);
}

///////////////////////////////////////////////////////////////////////////////////////
/// PARAMETRIC CHOLESKY NORMAL DISTRIBUTION
///////////////////////////////////////////////////////////////////////////////////////

ParametricCholeskyNormal::ParametricCholeskyNormal(const vec& initial_mean, const mat& initial_cholA)
    : ParametricNormal(initial_mean.n_elem)
{
    mean = initial_mean;
    cholCov = initial_cholA;
    this->updateInternalState();
}

void ParametricCholeskyNormal::wmle(const arma::vec& weights, const arma::mat& samples)
{
    double sumD = sum(weights);
    double sumD2 = sum(square(weights));
    double Z = sumD - sumD2/sumD;

    mean = samples*weights/sumD;

    arma::mat delta = samples;
    delta.each_col() -= mean;

    cholCov = arma::chol(delta*diagmat(weights)*delta.t())/std::sqrt(Z);

    updateInternalState();
}

vec ParametricCholeskyNormal::difflog(const vec& point) const
{
    int paramSize = this->getParametersSize();
    vec gradient(paramSize);

    //--- mean gradient
    vec diff = point - mean;
    vec mean_grad = invCov * diff;
    //---

    //--- Covariance gradient
    mat tmp = (point - mean) * (point - mean).t() * invCov;
    mat R = solve(cholCov.t(), tmp);
    mat dlogpdt_sigma(pointSize, pointSize, fill::zeros);
    for (int i = 0; i < pointSize; ++i)
        for (int j = i; j < pointSize; ++j)
            if (i == j)
                dlogpdt_sigma(i,j) = R(i,j) - 1.0 / cholCov(i,j);
            else
                dlogpdt_sigma(i,j) = R(i,j);

//    //TODO [OPTIMIZATION] fare meglio
//    mat idxs = trimatu(ones(pointSize, pointSize));
//    vec vals = dlogpdt_sigma.elem( find(idxs == 1) );
//    //---


    for (unsigned int i = 0; i < pointSize; ++i)
    {
        gradient[i] = mean_grad(i);
    }
//    for (int i = pointSize; i < paramSize; ++i)
//    {
//        gradient[i] = vals[i-pointSize];
//    }
    int rowi = 0, coli = 0;
    for (unsigned i = pointSize; i < paramSize; ++i)
    {
        gradient[i] = dlogpdt_sigma(rowi,coli);
        coli++;
        if (coli == pointSize)
        {
            rowi++;
            coli = rowi;
        }
    }
    return gradient;
}

mat ParametricCholeskyNormal::diff2log(const vec &point) const
{
    //TODO [IMPORTANT] implement
    return mat();
}

sp_mat ParametricCholeskyNormal::FIM() const
{
    //TODO [OPTIMIZATION] make in a more efficient way
    int rows = invCov.n_rows;
    int cols = invCov.n_cols;
    vector<mat> diag_blocks;
    diag_blocks.push_back(invCov);
    for (int k = 0, end_v = pointSize - 1; k < pointSize; ++k)
    {
        mat tmp = invCov( span(k, end_v), span(k, end_v) );
        tmp(0,0) += 1.0 / (cholCov(k,k)*cholCov(k,k));
        rows += tmp.n_rows;
        cols += tmp.n_cols;
        diag_blocks.push_back(tmp);
    }
    //==========================
    return blockdiagonal(diag_blocks, rows, cols);
    //=========================
}

sp_mat ParametricCholeskyNormal::inverseFIM() const
{
    //TODO [OPTIMIZATION] make in a more efficient way
    int rows = Cov.n_rows;
    int cols = Cov.n_cols;
    vector<mat> diag_blocks;
    diag_blocks.push_back(Cov);
    //std::cout << invCov << std::endl << std::endl;
    //std::cout << cholCov << std::endl << std::endl;
    for (int k = 0, end_v = pointSize - 1; k < pointSize; ++k)
    {
        mat tmp = invCov( span(k, end_v), span(k, end_v) );
        tmp(0,0) += 1.0 / (cholCov(k,k)*cholCov(k,k));
        mat nMtx = arma::solve(tmp.t(), arma::eye(tmp.n_rows, tmp.n_cols)).t(); //in matlab it is mrdivide operator
        //std::cout << nMtx << std::endl << std::endl;
        rows += nMtx.n_rows;
        cols += nMtx.n_cols;
        diag_blocks.push_back(nMtx);
    }
    //==========================
    return blockdiagonal(diag_blocks, rows, cols);
    //=========================
}

void ParametricCholeskyNormal::writeOnStream(ostream &out)
{
    int paramSize = this->getParametersSize();
    out << "ParametricCholeskyNormal" << std::endl;
    out << pointSize << std::endl;
    out << paramSize << std::endl;
    for (unsigned i = 0; i < paramSize; ++i)
    {
        out << mean(i) << "\t";
    }

    for (unsigned i = 0, ie = cholCov.n_elem; i < ie; ++i)
    {
        out << cholCov(i) << "\t";
    }

    out << std::endl;
}

void ParametricCholeskyNormal::readFromStream(istream &in)
{
    double val;
    in >> pointSize;
    int paramSize;
    in >> paramSize;

    // allocate space
    mean.set_size(pointSize);
    Cov.eye(pointSize,pointSize);
    invCov.eye(pointSize,pointSize);
    cholCov.zeros(pointSize,pointSize);

    for (unsigned i = 0; i < pointSize; ++i)
    {
        in >> val;
        mean(i) = val;
    }
    for (unsigned i = 0, ie = pointSize*pointSize; i < ie; ++i)
    {
        in >> val;
        cholCov(i) = val;
    }
    this->updateInternalState();
}

unsigned int ParametricCholeskyNormal::getParametersSize() const
{
    return 2*pointSize + (pointSize * pointSize - pointSize) / 2;
}

arma::vec ParametricCholeskyNormal::getParameters() const
{
    int dim = getParametersSize();
    vec params(dim);
    for (int i = 0; i < pointSize; ++i)
        params[i] = mean[i];


    int rowi = 0, coli = 0;
    for (unsigned i = 0, ie = dim-pointSize; i < ie; ++i)
    {
        params(pointSize+i) = cholCov(rowi,coli);
        coli++;
        if (coli == pointSize)
        {
            rowi++;
            coli = rowi;
        }
    }
    return params;
}

void ParametricCholeskyNormal::setParameters(const arma::vec& newval)
{
    int dim = getParametersSize();
    for (int i = 0; i < pointSize; ++i)
        mean[i] = newval[i];


    int rowi = 0, coli = 0;
    for (unsigned i = 0, ie = dim-pointSize; i < ie; ++i)
    {
        cholCov(rowi,coli) = newval(pointSize+i);
        coli++;
        if (coli == pointSize)
        {
            rowi++;
            coli = rowi;
        }
    }
    updateInternalState();
}

void ParametricCholeskyNormal::update(const arma::vec &increment)
{
    int dim = getParametersSize();
    for (int i = 0; i < pointSize; ++i)
        mean[i] += increment[i];


    int rowi = 0, coli = 0;
    for (unsigned i = 0, ie = dim-pointSize; i < ie; ++i)
    {
        cholCov(rowi,coli) += increment(pointSize+i);
        coli++;
        if (coli == pointSize)
        {
            rowi++;
            coli = rowi;
        }
    }
    updateInternalState();
}

void ParametricCholeskyNormal::updateInternalState()
{
    Cov = cholCov.t() * cholCov;
    //TODO [OPTIMIZATION] questo si potrebbe fare meglio
    invCov = inv(Cov);
    detValue = det(Cov);
}

///////////////////////////////////////////////////////
/// Full covariance matrix
///////////////////////////////////////////////////////
ParametricFullNormal::ParametricFullNormal(const arma::vec& initial_mean,
        const arma::mat& initial_cov) : ParametricNormal(initial_mean.n_elem)
{
    mean = initial_mean;
    Cov = initial_cov;

    updateInternalState();
}

arma::vec ParametricFullNormal::difflog(const arma::vec& point) const
{
    //TODO [IMPORTANT] implement
    return mat();
}

arma::mat ParametricFullNormal::diff2log(const arma::vec& point) const
{
    //TODO [IMPORTANT] implement
    return mat();
}

arma::sp_mat ParametricFullNormal::FIM() const
{
    //TODO [IMPORTANT] implement
    return sp_mat();
}

arma::sp_mat ParametricFullNormal::inverseFIM() const
{
    //TODO [IMPORTANT] implement
    return sp_mat();
}

void ParametricFullNormal::wmle(const arma::vec& weights, const arma::mat& samples)
{
    double sumD = sum(weights);
    double sumD2 = sum(square(weights));
    double Z = sumD - sumD2/sumD;

    mean = samples*weights/sumD;

    arma::mat delta = samples;
    delta.each_col() -= mean;
    Cov = delta*diagmat(weights)*delta.t()/Z;

    updateInternalState();
}

void ParametricFullNormal::writeOnStream(std::ostream &out)
{
    //TODO [SERIALIZATION] implement
}

void ParametricFullNormal::readFromStream(std::istream &in)
{
    //TODO [SERIALIZATION] implement
}

unsigned int ParametricFullNormal::getParametersSize() const
{
    return mean.n_elem + Cov.n_elem;
}

arma::vec ParametricFullNormal::getParameters() const
{
    arma::vec w(getParametersSize());
    w.rows(0, mean.n_elem - 1) = mean;
    w.rows(mean.n_elem, w.n_elem - 1) = vectorise(Cov);
    return w;
}

void ParametricFullNormal::setParameters(const arma::vec& newval)
{
    mean = newval.rows(0, mean.n_elem - 1);

    const auto& newCov = newval.rows(mean.n_elem, newval.n_elem -1);

    for(unsigned i = 0; i < newCov.n_elem; i++)
        Cov[i] = newCov[i];


    updateInternalState();
}

void ParametricFullNormal::update(const arma::vec &increment)
{
    mean += increment.rows(0, mean.n_elem - 1);

    const auto& covInc = increment.rows(mean.n_elem, increment.n_elem -1);

    for(unsigned i = 0; i < covInc.n_elem; i++)
        Cov[i] += covInc[i];

    updateInternalState();

}


void ParametricFullNormal::updateInternalState()
{
    invCov     = inv(Cov);
    detValue   = det(Cov);
    cholCov    = chol(Cov);
}



} //end namespace
