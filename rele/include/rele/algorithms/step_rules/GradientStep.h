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

#ifndef STEPRULES_H_
#define STEPRULES_H_

#include <iostream>
#include <armadillo>

namespace ReLe
{

/*!
 * This interface implements a generic gradient step rule, i.e. a rule
 * to select a parameter update, given the current parameters gradient
 * (and optionally other information).
 */
class GradientStep
{
public:
    /*!
     * Computes the new parameters step assuming identity metric.
     * \param gradient the actual gradient
     * \return the delta to apply on the parameters
     */
    virtual arma::vec operator()(const arma::vec& gradient) = 0;

    /*!
     * Computes the new parameters step using the gradient and
     * the natural gradient.
     * \param gradient the vanilla gradient
     * \param nat_gradient the natural gradient
     * \return the delta to apply on the parameters
     */
    virtual arma::vec operator()(const arma::vec& gradient,
                                 const arma::vec& nat_gradient) = 0;

    /*!
     * Computes the new parameters step assuming the given metric.
     * \param gradient the gradient direction
     * \param metric a predefined space metric
     * \param inverse whether the metric parameter is the inverse (\f$M^{-1}\f$) one or not (\f$M\f$)
     * \return the delta to apply on the parameters
     */
    virtual arma::vec operator()(const arma::vec& gradient,
                                 const arma::mat& metric,
                                 bool inverse) = 0;

    /*!
     * This function is called in order to reset the internal state of the class
     */
    virtual void reset() = 0;

    virtual ~GradientStep()
    {

    }

protected:
    /*!
     * Default function for computing the product:
     * \f[ \ M^{-1}\nabla_{\theta}J \f]
     *
     * when \f$M=\mathcal{F}\f$, the Fisher Information Matrix,
     * this function computes the natural gradient
     *
     * \param gradient the vanilla gradient
     * \param metric a predefined space metric
     * \inverse whether the metric parameter is the inverse (\f$M^{-1}\f$) one or not (\f$M\f$)
     * \return the product \f$ \ M^{-1}\nabla_{\theta}J \f$
     *
     */
    arma::vec computeGradientInMetric(const arma::vec& gradient, const arma::mat& metric, bool inverse)
    {
        arma::vec g;
        if(inverse)
        {
            g = metric*gradient;
        }
        else
        {
        	if(arma::rank(metric) == metric.n_rows)
        		g = arma::solve(metric, gradient);
        	else
        		g = arma::pinv(metric)*gradient;
        }

        return g;
    }

};

/*!
 * Basic step rule.
 * The step is very simple:
 * \f[ \Delta\theta = \alpha\nabla_{\theta}J \f]
 */
class ConstantGradientStep : public GradientStep
{
public:
    /*!
     * Constructor.
     * \param alpha the constant factor to multiply to the gradient.
     */
    ConstantGradientStep(double alpha): alpha(alpha)
    {

    }

    virtual arma::vec operator()(const arma::vec& gradient) override
    {
        return alpha*gradient;
    }

    virtual arma::vec operator()(const arma::vec& gradient,
                                 const arma::vec& nat_gradient) override
    {
        auto& self = *this;
        return self(nat_gradient);
    }

    virtual arma::vec operator()(const arma::vec& gradient,
                                 const arma::mat& metric,
                                 bool inverse)  override
    {
        auto& self = *this;
        return self(computeGradientInMetric(gradient, metric, inverse));
    }



    void reset() override
    {
    }

protected:
    double alpha;
};

/*!
 * A constant vectorial step rule.
 * The step is very simple:
 * \f[ \Delta\theta = \alpha\odot\nabla_{\theta}J \f]
 * with:
 * \f[ \alpha,\theta\in\mathbb{R}^n \f]
 */
class VectorialGradientStep: public GradientStep
{
public:
    /*!
     * Constructor.
     * \param alpha the vector of factors to multiply to
     * each component of the gradient.
     */
    VectorialGradientStep(const arma::vec& alpha): alpha(alpha)
    {

    }

    virtual arma::vec operator()(const arma::vec& gradient) override
    {
        return alpha%gradient;
    }

    virtual arma::vec operator()(const arma::vec& gradient,
                                 const arma::vec& nat_gradient) override
    {
        auto& self = *this;
        return self(nat_gradient);
    }

    virtual arma::vec operator()(const arma::vec& gradient,
                                 const arma::mat& metric,
                                 bool inverse)  override
    {
        auto& self = *this;
        return self(computeGradientInMetric(gradient, metric, inverse));
    }



    void reset() override
    {
    }

protected:
    arma::vec alpha;
};

/*!
 * This class implements a basic adaptative gradient step.
 * Instead of moving of a step proportional to the gradient,
 * takes a step limited by a given metric.
 * If no metric is given, the identity matrix is used.
 *
 * The step rule is:
 * \f[
 * \Delta\theta=\underset{\Delta\vartheta}{argmax}\Delta\vartheta^{t}\nabla_{\theta}J
 * \f]
 *
 * \f[
 * s.t.:\Delta\vartheta^{T}M\Delta\vartheta\leq\varepsilon
 * \f]
 *
 * References
 * ==========
 *
 * [Neumann. Lecture Notes](http://www.ias.informatik.tu-darmstadt.de/uploads/Geri/lecture-notes-constraint.pdf)
 */
class AdaptiveGradientStep : public GradientStep
{
public:
    /*!
     * Constructor.
     * \param eps the maximum allowed size for the step.
     */
    AdaptiveGradientStep(double eps): stepValue(eps)
    {
    }

    virtual arma::vec operator()(const arma::vec& gradient) override
    {
        auto& self = *this;
        return self(gradient, gradient);
    }

    virtual arma::vec operator()(const arma::vec& gradient,
                                 const arma::vec& nat_gradient) override
    {
        double tmp = arma::as_scalar(gradient.t() * nat_gradient);
        double lambda = sqrt(tmp / (4 * stepValue));
        lambda = std::max(lambda, 1e-8); // to avoid numerical problems
        double step_length = 1.0 / (2.0 * lambda);

        return step_length*nat_gradient;
    }

    virtual arma::vec operator()(const arma::vec& gradient,
                                 const arma::mat& metric,
                                 bool inverse)  override
    {
        auto& self = *this;
        arma::mat nat_grad = computeGradientInMetric(gradient, metric, inverse);
        return self(gradient, nat_grad);
    }


    void reset() override
    {
    }

protected:
    double stepValue;
};

} //end namespace

#endif //STEPRULES_H_
