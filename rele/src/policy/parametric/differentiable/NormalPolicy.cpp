#include "parametric/differentiable/NormalPolicy.h"
#include "RandomGenerator.h"

using namespace std;

namespace ReLe
{

///////////////////////////////////////////////////////////////////////////////////////
/// NORMAL POLICY
///////////////////////////////////////////////////////////////////////////////////////

void NormalPolicy::calculateMeanAndStddev(const arma::vec& state)
{
    // compute mean value
    arma::vec mean = (*approximator)(state);
    mMean = mean[0];
}

double NormalPolicy::operator()(const arma::vec& state, const arma::vec& action)
{

    // compute mean value
    calculateMeanAndStddev(state);

    double scalara = action[0];

    // compute probability
    return ReLe::normpdf(scalara, mMean, mInitialStddev*mInitialStddev);
}

arma::vec NormalPolicy::operator() (const arma::vec& state)
{
    // compute mean value
    calculateMeanAndStddev(state);
    //        MY_PRINT(mMean);
    //        MY_PRINT(mInitialStddev);

    double normn = RandomGenerator::sampleNormal();
    //        MY_PRINT(normn);

    arma::vec output(1);
    output[0] = normn * mInitialStddev + mMean;
    return output;
}

arma::vec NormalPolicy::diff(const arma::vec& state, const arma::vec& action)
{
    return (*this)(state,action) * difflog(state,action);
}

arma::vec NormalPolicy::difflog(const arma::vec& state, const arma::vec& action)
{
    // get scalar action
    double a = action[0];

    // compute mean value
    calculateMeanAndStddev(state);

    // get features
    AbstractBasisMatrix& basis = approximator->getBasis();
    arma::mat features = basis(state);

    // compute gradient
    return features * (a - mMean) / (mInitialStddev * mInitialStddev);
}

arma::mat NormalPolicy::diff2log(const arma::vec& state, const arma::vec& action)
{

    // compute mean value
    calculateMeanAndStddev(state);

    // get features
    AbstractBasisMatrix& basis = approximator->getBasis();
    arma::mat features = basis(state);

    return -features * features.t() / (mInitialStddev * mInitialStddev);

}


///////////////////////////////////////////////////////////////////////////////////////
/// NORMAL POLICY WITH STATE DEPENDANT STDDEV (STD is not a parameter to be learned)
///////////////////////////////////////////////////////////////////////////////////////
void
NormalStateDependantStddevPolicy::calculateMeanAndStddev(const arma::vec& state)
{
    // compute mean value
    arma::vec mean = (*approximator)(state);
    mMean = mean[0];

    // compute stddev
    arma::vec evalstd = (*stdApproximator)(state);
    mInitialStddev = evalstd[0];
}


///////////////////////////////////////////////////////////////////////////////////////
/// MVN POLICY
///////////////////////////////////////////////////////////////////////////////////////

double MVNPolicy::operator()(const arma::vec& state, const arma::vec& action)
{
    UpdateInternalState(state);
    std::cout << mMean.t();
    std::cout << mCinv << std::endl;
    std::cout << mDeterminant << std::endl;
    return mvnpdfFast(action, mMean, mCinv, mDeterminant);
}

arma::vec MVNPolicy::operator() (const arma::vec& state)
{
    UpdateInternalState(state);
    return mvnrandFast(mMean, mCholeskyDec);
}

arma::vec MVNPolicy::diff(const arma::vec &state, const arma::vec &action)
{
    return (*this)(state,action) * difflog(state,action);
}

arma::vec MVNPolicy::difflog(const arma::vec &state, const arma::vec &action)
{
    UpdateInternalState(state);

    arma::vec smdiff(mCovariance.n_rows);
    //        std::cout << "Action: ";
    for (unsigned i = 0; i < mCovariance.n_rows; ++i)
    {
        smdiff(i) = action[i] - mMean(i);
        //            std::cout << action[i] << " ";
    }
    //        std::cout << "\n";

    AbstractBasisMatrix& basis = approximator->getBasis();
    arma::mat features = basis(state);

    //        MY_PRINT(features);
    //        MY_PRINT(smdiff);
    //        MY_PRINT(mCinv);

    // compute gradient
    return 0.5 * features * (mCinv + mCinv.t()) * smdiff;
}

arma::mat MVNPolicy::diff2log(const arma::vec &state, const arma::vec &action)
{
    UpdateInternalState(state);

    AbstractBasisMatrix& basis = approximator->getBasis();
    arma::mat features = basis(state);

    // compute gradient
    return - 0.5 * features * (mCinv + mCinv.t()) * features.t();
}

///////////////////////////////////////////////////////////////////////////////////////
/// MVN POLICY with Diagonal covariance (parameters of the diagonal are stddev)
///////////////////////////////////////////////////////////////////////////////////////

void MVNDiagonalPolicy::setParameters(arma::vec &w)
{
    assert(w.n_elem == this->getParametersSize());
    int dp = approximator->getParametersSize();
    arma::vec tmp = w.rows(0, dp-1);
    approximator->setParameters(tmp);
    for (int i = 0, ie = stddevParams.n_elem; i < ie; ++i)
    {
        stddevParams(i) = w[dp + i];
        assert(!std::isnan(stddevParams(i)) && !std::isinf(stddevParams(i)));
    }
    UpdateCovarianceMatrix();
}

arma::vec MVNDiagonalPolicy::difflog(const arma::vec &state, const arma::vec &action)
{
    UpdateInternalState(state);

    arma::vec smdiff(mCovariance.n_rows);
    arma::vec gradstddev(mCovariance.n_rows);
    //        std::cout << "Action: ";
    for (unsigned i = 0; i < mCovariance.n_rows; ++i)
    {
        smdiff(i) = action[i] - mMean(i);
        double val = smdiff[i];
        gradstddev[i] = - 1.0 / stddevParams(i) + (val * val) /  (stddevParams(i) * stddevParams(i) * stddevParams(i));
    }
    //        std::cout << "\n";

    AbstractBasisMatrix& basis = approximator->getBasis();
    arma::mat features = basis(state);
    //    std::cerr << "action: " << action;
    //    std::cerr << "features: " << features;
    //    std::cerr << "mean: " << mMean.t();
    //    std::cerr << "stddev: " << stddevParams.t();

    //        MY_PRINT(features);
    //        MY_PRINT(smdiff);
    //        MY_PRINT(mCinv);

    // compute gradient
    arma::vec gradm = 0.5 * features * (mCinv + mCinv.t()) * smdiff;
    return arma::join_vert(gradm, gradstddev);
}

arma::mat MVNDiagonalPolicy::diff2log(const arma::vec &state, const arma::vec &action)
{
    UpdateInternalState(state);

    //TODO controllare implementazione
    int paramSize = this->getParametersSize();
    arma::mat hessian(paramSize,paramSize,arma::fill::zeros);
    int dm = approximator->getParametersSize();
    int ds = stddevParams.n_elem;


    AbstractBasisMatrix& basis = approximator->getBasis();
    arma::mat features = basis(state);
    arma::mat Hm = - 0.5 * features * (mCinv + mCinv.t()) * features.t();

    for (unsigned i = 0; i < dm; ++i)
        for (unsigned k = 0; k < dm; ++k)
        {
            // d2 logp / dmdm
            hessian(i,k) = Hm(i,k);
        }

    for (unsigned i = 0; i < ds; ++i)
    {
        int idx = dm + i;
        double diff = action[i] - mMean(i);

        // consider only the components different from zero
        // obtained from the derivative of the gradient w.r.t. the covariance parameters
        // these terms are only mSupportSize
        // d2 logp / dpdp
        double sigma2 = stddevParams(i) * stddevParams(i);
        hessian(idx,idx) = 1.0 / (sigma2) - 3.0 * (diff * diff) /  (sigma2 * sigma2);;
    }

    for (unsigned m = 0; m < dm; ++m)
    {
        for (unsigned k = 0; k < ds; ++k)
        {
            double diff = action[k] - mMean(k);
            double sigma2 = stddevParams(k) * stddevParams(k);
            // d2 logp / dpdm
            hessian(dm+k, m) = - 2.0 * features(m,k) * diff / (sigma2 * stddevParams(k));

            // d2 logp / dmdp
            hessian(m, dm+k) = hessian(dm+k, m);
        }
    }
    return hessian;
}

void MVNDiagonalPolicy::UpdateCovarianceMatrix()
{
    mDeterminant = 1.0;
    for (unsigned int i = 0; i < stddevParams.n_elem; ++i)
    {
        mCovariance(i,i) = stddevParams(i) * stddevParams(i);
        mCinv(i,i) = 1.0 / mCovariance(i,i);
        // check that the covariance is not NaN or Inf
        assert(!std::isnan(mCovariance(i,i)) && !std::isinf(mCovariance(i,i)));
        assert(!std::isnan(mCinv(i,i)) && !std::isinf(mCinv(i,i)));
        mDeterminant *= mCovariance(i,i);
        mCholeskyDec(i,i) = sqrt(mCovariance(i,i));
    }
//    mCholeskyDec = arma::chol(mCovariance);
}

///////////////////////////////////////////////////////////////////////////////////////
/// MVNLogisticPolicy
///////////////////////////////////////////////////////////////////////////////////////

arma::vec MVNLogisticPolicy::difflog(const arma::vec& state, const arma::vec& action)
{
    UpdateInternalState(state);
    arma::vec smdiff(mCovariance.n_rows);
    for (unsigned i = 0; i < mCovariance.n_rows; ++i)
    {
        smdiff(i) = action[i] - mMean(i);
    }
    AbstractBasisMatrix& basis = approximator->getBasis();
    arma::mat features = basis(state);

    // compute gradient
    arma::vec meang = 0.5 * features * (mCinv + mCinv.t()) * smdiff;
    unsigned int dim = meang.n_elem, gradDim = dim + mLogisticParams.n_elem;

    //correct gradient dimension
    arma::vec gradient(gradDim, arma::fill::zeros);

    //first fill gradient of the mean
    for (unsigned int i = 0; i < dim; ++i)
    {
        gradient(i) = meang(i);
    }

    //gradient of covariance matrix
    for (unsigned int i = 0, ie = mLogisticParams.n_elem; i < ie; ++i)
    {
        double logisticVal = logistic(mLogisticParams(i), mAsVariance(i));
        double A = - 0.5 * logisticVal * exp(-mLogisticParams(i)) / mAsVariance(i);
        double B = 0.5 * exp(-mLogisticParams(i))
                   * smdiff(i) * smdiff(i) / mAsVariance(i);
        //        std::cout <<"A: " << A << std::endl;
        //        std::cout <<"B: " << B << std::endl;
        gradient(i+dim) = A + B;
    }
    return gradient;
}

arma::mat MVNLogisticPolicy::diff2log(const arma::vec& state, const arma::vec& action)
{
    return arma::mat();
}

void MVNLogisticPolicy::UpdateCovarianceMatrix()
{

    mDeterminant = 1.0;
    for (unsigned int i = 0; i < mLogisticParams.n_elem; ++i)
    {
        mCovariance(i,i) = logistic(mLogisticParams(i), mAsVariance(i));
        mCinv(i,i) = 1.0 / mCovariance(i,i);
        // check that the covariance is not NaN or Inf
        assert(!std::isnan(mCovariance(i,i)) && !std::isinf(mCovariance(i,i)));
        assert(!std::isnan(mCinv(i,i)) && !std::isinf(mCinv(i,i)));
        mDeterminant *= mCovariance(i,i);
        mCholeskyDec(i,i) = sqrt(mCovariance(i,i));
    }
//        mCholeskyDec = arma::chol(mCovariance);
}


} //end namespace
