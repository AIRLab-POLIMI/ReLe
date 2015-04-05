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

class StepRule
{
public:
    /**
     * @brief computes the new step length given the information
     * @param gradient the actual gradient
     * @param metric a predefined space metric
     * @return
     */
    virtual arma::vec stepLength(arma::vec& gradient, arma::mat& metric, bool inverse = false) = 0;

    /**
     * This function is called in order to reset the internal state of the class
     * @brief reset the internal state of class
     */
    virtual void reset() = 0;
};

class ConstantStep : public StepRule
{
public:
    ConstantStep(double val): stepDirection(1)
    {
        stepDirection(0) = val;
    }
    ConstantStep(arma::vec val): stepDirection(val)
    {
    }

    inline arma::vec stepLength(arma::vec& gradient, arma::mat& metric, bool inverse = false)
    {
        return stepDirection;
    }

    void reset()
    {
    }

protected:
    arma::vec stepDirection;
};

class AdaptiveStep : public StepRule
{
public:
    AdaptiveStep(double val): stepValue(val)
    {
    }

    inline arma::vec stepLength(arma::vec& gradient, arma::mat& metric, bool inverse = false)
    {
        double lambda, step_length;
        if (inverse == true)
        {
            arma::mat tmp = gradient.t() * (metric * gradient);
            lambda = sqrt(tmp(0,0) / (4 * stepValue));
            lambda = std::max(lambda, 1e-8); // to avoid numerical problems
            step_length = 1.0 / (2.0 * lambda);
        }
        else
        {
            //--- Compute learning step
            //http://www.ias.informatik.tu-darmstadt.de/uploads/Geri/lecture-notes-constraint.pdf
            arma::mat tmp;
            int rnk = arma::rank(metric);
            //        std::cout << rnk << " " << fisher << std::endl;
            if (rnk == metric.n_rows)
            {
                arma::mat H = arma::solve(metric, gradient);
                tmp = gradient.t() * H;
                lambda = sqrt(tmp(0,0) / (4 * stepValue));
                lambda = std::max(lambda, 1e-8); // to avoid numerical problems
                step_length = 1.0 / (2.0 * lambda);
            }
            else
            {
                arma::mat H = arma::pinv(metric);
                tmp = gradient.t() * (H * gradient);
                lambda = sqrt(tmp(0,0) / (4 * stepValue));
                lambda = std::max(lambda, 1e-8); // to avoid numerical problems
                step_length = 1.0 / (2.0 * lambda);
            }
            //---
        }
        arma::vec output(1);
        output(0) = step_length;
        return output;
    }

    void reset()
    {
    }

protected:
    double stepValue;
};

} //end namespace

#endif //STEPRULES_H_
