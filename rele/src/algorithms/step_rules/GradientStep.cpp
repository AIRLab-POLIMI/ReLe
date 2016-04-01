/*
 * rele,
 *
 *
 * Copyright (C) 2016 Davide Tateo & Matteo Pirotta
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
#include "rele/algorithms/step_rules/GradientStep.h"

using namespace arma;

namespace ReLe
{

vec GradientStep::computeGradientInMetric(const vec& gradient, const mat& metric, bool inverse)
{
    vec g;
    if(inverse)
    {
        g = metric*gradient;
    }
    else
    {
        if(rank(metric) == metric.n_rows)
            g = solve(metric, gradient);
        else
            g = pinv(metric)*gradient;
    }

    return g;
}

ConstantGradientStep::ConstantGradientStep(double alpha): alpha(alpha)
{

}

vec ConstantGradientStep::operator()(const vec& gradient)
{
    return alpha*gradient;
}

vec ConstantGradientStep::operator()(const vec& gradient,
                                     const vec& nat_gradient)
{
    auto& self = *this;
    return self(nat_gradient);
}

vec ConstantGradientStep::operator()(const vec& gradient,
                                     const mat& metric,
                                     bool inverse)
{
    auto& self = *this;
    return self(computeGradientInMetric(gradient, metric, inverse));
}



void ConstantGradientStep::reset()
{
}


VectorialGradientStep::VectorialGradientStep(const vec& alpha): alpha(alpha)
{

}

vec VectorialGradientStep::operator()(const vec& gradient)
{
    return alpha%gradient;
}

vec VectorialGradientStep::operator()(const vec& gradient,
                                      const vec& nat_gradient)
{
    auto& self = *this;
    return self(nat_gradient);
}

vec VectorialGradientStep::operator()(const vec& gradient,
                                      const mat& metric,
                                      bool inverse)
{
    auto& self = *this;
    return self(computeGradientInMetric(gradient, metric, inverse));
}



void VectorialGradientStep::reset()
{
}

AdaptiveGradientStep::AdaptiveGradientStep(double eps): stepValue(eps)
{
}

vec AdaptiveGradientStep::operator()(const vec& gradient)
{
    auto& self = *this;
    return self(gradient, gradient);
}

vec AdaptiveGradientStep::operator()(const vec& gradient,
                                     const vec& nat_gradient)
{
    double tmp = as_scalar(gradient.t() * nat_gradient);
    double lambda = sqrt(tmp / (4 * stepValue));
    lambda = std::max(lambda, 1e-8); // to avoid numerical problems
    double step_length = 1.0 / (2.0 * lambda);

    return step_length*nat_gradient;
}

vec AdaptiveGradientStep::operator()(const vec& gradient,
                                     const mat& metric,
                                     bool inverse)
{
    auto& self = *this;
    mat nat_grad = computeGradientInMetric(gradient, metric, inverse);
    return self(gradient, nat_grad);
}


void AdaptiveGradientStep::reset()
{
}


}
