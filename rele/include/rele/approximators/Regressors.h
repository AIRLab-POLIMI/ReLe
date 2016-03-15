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

#ifndef REGRESSORS_H_
#define REGRESSORS_H_

#include "rele/core/Basics.h"
#include "rele/approximators/data/BatchData.h"
#include "rele/approximators/Features.h"

namespace ReLe
{

/*!
 * This is the default interface for function approximators.
 * We suppose that function approximation works applying an arbitrary function over
 * a set of features of the input data.
 * Formally a regressor is a function \f$f(i)\rightarrow\mathcal{D}_{output}^n\f$ with \f$i\in\mathcal{D}_{input}\f$
 * Function approximation can work with any type of input data and features over data (dense, sparse).
 */
template<class InputC, class OutputC, bool denseOutput = true>
class Regressor_
{

public:

    /*!
     * Constructor.
     * \param phi the features used by the approximator
     * \param output the dimensionality of the output vector
     */
    Regressor_(Features_<InputC, denseOutput>& phi, unsigned int output = 1) :
        phi(phi), outputDimension(output)
    {
    }


    /*!
     * Evaluates the function at the input.
     * \param input the input data
     * \return the value of the function at input
     */
    virtual OutputC operator() (const InputC& input) = 0;

    /*!
     * Overloading of the evaluation method, simply vectorizes the inputs and then evaluates
     * the regressor with the default 1 input method
     * \param input1 the first input data
     * \param input2 the second input data
     * \return the value of the function
     */
    template<class Input1, class Input2>
    OutputC operator()(const Input1& input1, const Input2& input2)
    {
        auto& self = *this;
        return self(vectorize(input1, input2));
    }

    /*!
     * Overloading of the evaluation method, simply vectorizes the inputs and then evaluates
     * the regressor with the default 1 input method
     * \param input1 the first input data
     * \param input2 the second input data
     * \param input3 the third input data
     * \return the value of the function
     */
    template<class Input1, class Input2, class Input3>
    OutputC operator()(const Input1& input1, const Input2& input2, const Input3& input3)
    {
        auto& self = *this;
        return self(vectorize(input1, input2, input3));
    }

    /*!
     * Getter.
     * \return the dimensionality of the output.
     */
    int getOutputSize()
    {
        return outputDimension;
    }

    /*!
     * Return the features used by the regressor.
     */
    inline Features_<InputC, denseOutput>& getFeatures()
    {
        return phi;
    }

    /*!
     * Destructor.
     */
    virtual ~Regressor_()
    {
    }

protected:
    Features_<InputC, denseOutput>& phi;
    unsigned int outputDimension;
};

typedef Regressor_<arma::vec, arma::vec> Regressor;

/*!
 * This is the default interface for function approximators that can be described through parameters.
 * This interface assumes that the parametrization should be differentiable.
 */
template<class InputC, bool denseOutput = true>
class ParametricRegressor_: public Regressor_<InputC, arma::vec, denseOutput>
{
public:
    /*!
     * Constructor.
     * \param phi the features used by the approximator
     * \param output the dimensionality of the output vector
     */
    ParametricRegressor_(Features_<InputC, denseOutput>& phi, unsigned int output = 1) :
        Regressor_<InputC, arma::vec, denseOutput>(phi, output)
    {
    }

    /*!
     * Setter.
     * \param params the parameters vector
     */
    virtual void setParameters(const arma::vec& params) = 0;

    /*!
     * Getter.
     * \return the parameters vector
     */
    virtual arma::vec getParameters() const = 0;

    /*!
     * Getter.
     * \return the length of parameters vector
     */
    virtual unsigned int getParametersSize() const = 0;

    /*!
     * Compute the derivative of the represented function w.r.t. the parameters,
     * at input.
     */
    virtual arma::vec diff(const InputC& input) = 0;

    /*!
     * Overloading of the differentiation method, simply vectorizes the inputs and then evaluates
     * the derivative with the default 1 input method
     * \param input1 the first input data
     * \param input2 the second input data
     * \return the value of the derivative
     */
    template<class Input1, class Input2>
    arma::vec diff(const Input1& input1, const Input2& input2)
    {
        return this->diff(vectorize(input1, input2));
    }

    /*!
     * Overloading of the differentiation method, simply vectorizes the inputs and then evaluates
     * the derivative with the default 1 input method
     * \param input1 the first input data
     * \param input2 the second input data
     * \param input3 the third input data
     * \return the value of the derivative
     */
    template<class Input1, class Input2, class Input3>
    arma::vec diff(const Input1& input1, const Input2& input2, const Input3& input3)
    {
        return this->diff(vectorize(input1, input2, input3));
    }

    /*!
     * Destructor.
     */
    virtual ~ParametricRegressor_()
    {

    }

};

typedef ParametricRegressor_<arma::vec> ParametricRegressor;

/*!
 * This is the default interface for regressors that uses supervised learning for learning the target function.
 * It contains functions to train the regressor from raw data or from already computed features,
 * and performance evaluation methods as well.
 */
template<class InputC, class OutputC, bool denseOutput=true>
class BatchRegressor_ : public Regressor_<InputC, OutputC, denseOutput>
{
    typedef typename input_traits<denseOutput>::type FeaturesCollection;
    typedef typename output_traits<OutputC>::type OutputCollection;
public:
    /*!
     * Constructor.
     * \param phi the features used by the approximator
     * \param output the dimensionality of the output vector
     */
    BatchRegressor_(Features_<InputC, denseOutput>& phi, unsigned int output = 1) :
        Regressor_<InputC, OutputC, denseOutput>(phi, output)
    {

    }

    /*!
     * This method is used to train raw input data
     * \param dataset the input dataset
     */
    virtual void train(const BatchDataRaw_<InputC, OutputC>& dataset)
    {
        unsigned int N = dataset.size();

        //compute features matrix
        FeaturesCollection features(this->phi.rows(), N);
        OutputCollection outputs(this->outputDimension, N);

        for(int i = 0; i < N; i++)
        {
            features.col(i) = this->phi(dataset.getInput(i));
            outputs.col(i) = dataset.getOutput(i);
        }

        BatchDataSimple_<OutputC, denseOutput> featureDataset(features, outputs);
        trainFeatures(featureDataset);
    }

    /*!
     * This method is used to compute the performance of an input dataset w.r.t.
     * the current learned regressor.
     * \param dataset the input dataset
     * \return the value of the objective function
     */
    double computeJ(const BatchDataRaw_<InputC, arma::vec>& dataset)
    {
        unsigned int N = dataset.size();

        //compute features matrix
        FeaturesCollection features(this->phi.rows(), N);
        OutputCollection outputs(this->outputDimension, N);

        for(int i = 0; i < N; i++)
        {
            features.col(i) = this->phi(dataset.getInput(i));
            outputs.col(i) = dataset.getOutput(i);
        }

        BatchDataSimple_<OutputC, denseOutput> featureDataset(features, outputs);

        return computeJFeatures(featureDataset);
    }

    /*!
     * This method implements the low level training of a dataset from a set of already computed features.
     * \param dataset the set of computed features.
     */
    virtual void trainFeatures(const BatchData_<OutputC, denseOutput>& dataset) = 0;

    /*!
     * This method is used to compute the performance of the features dataset w.r.t.
     * the current learned regressor.
     * \param dataset the features dataset
     * \return the value of the objective function
     */
    virtual double computeJFeatures(const BatchData_<OutputC, denseOutput>& dataset) = 0;


    /*!
     * Destructor.
     */
    virtual ~BatchRegressor_()
    {

    }

};

typedef BatchRegressor_<arma::vec, arma::vec> BatchRegressor;


/*!
 * This is the default interface for regressors that uses unsupervised learning for learning the target function.
 * It contains functions to train the regressor from raw data or from already computed feature.
 */
template<class InputC, class OutputC, bool denseOutput = true>
class UnsupervisedBatchRegressor_ : public Regressor_<InputC, OutputC, denseOutput>
{
    typedef typename input_traits<denseOutput>::type FeaturesCollection;
public:
    /*!
     * Constructor.
     * \param phi the features used by the approximator
     * \param output the dimensionality of the output vector
     */
    UnsupervisedBatchRegressor_(Features_<InputC, denseOutput>& phi, unsigned int output = 1)
        : Regressor_<InputC, OutputC, denseOutput>(phi, output)
    {

    }

    /*!
     * This method is used to compute the performance of an input dataset w.r.t.
     * the current learned regressor.
     * \param dataset the input dataset
     * \return the value of the objective function
     */
    virtual void train(const std::vector<InputC>& dataset)
    {
        unsigned int N = dataset.size();

        //compute features matrix
        arma::mat features(this->phi.rows(), N);
        for(int i = 0; i < N; i++)
        {
            features.col(i) = this->phi(dataset[i]);
        }

        trainFeatures(features);
    }

    /*!
     * This method implements the low level training of a dataset from a set of already computed features.
     * \param dataset the set of computed features.
     */
    virtual void trainFeatures(const FeaturesCollection& features) = 0;

    /*!
     * Destructor.
     */
    virtual ~UnsupervisedBatchRegressor_()
    {

    }

};

}

#define USE_REGRESSOR_MEMBERS(InputC, OutputC, denseOutput) \
    typedef Regressor_<InputC, OutputC, denseOutput> Base; \
    using Base::phi;

#define USE_PARAMETRIC_REGRESSOR_MEMBERS(InputC, OutputC, denseOutput) \
    typedef ParametricRegressor_<InputC, denseOutput> Base;

#define USE_BATCH_REGRESSOR_MEMBERS(InputC, OutputC, denseOutput) \
    typedef BatchRegressor_<InputC, OutputC, denseOutput> Base;

#define USE_UNSUPERVISED_REGRESSOR_MEMBERS(InputC, OutputC, denseOutput) \
    typedef UnsupervisedBatchRegressor_<InputC, OutputC, denseOutput> Base; \
 
#define DEFINE_FEATURES_TYPES(denseOutput) \
	typedef typename input_traits<denseOutput>::type FeaturesCollection; \
	typedef typename input_traits<denseOutput>::column_type FeaturesVector;

#endif /* REGRESSORS_H_ */
