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

#ifndef INCLUDE_RELE_IRL_UTILS_OPTIMIZATION_H_
#define INCLUDE_RELE_IRL_UTILS_OPTIMIZATION_H_

namespace ReLe
{

class Optimization
{
public:
    static void inequalitySimplexConstraints(unsigned m, double* result,
            unsigned n, const double* x, double* grad, void* f_data)
    {
        result[n] = -1.0;
        for (unsigned int i = 0; i < n; ++i)
        {
            // -x_i <= 0
            result[i] = -x[i];
            // sum x_i - 1 <= 0
            result[n] += x[i];
        }

        //compute the gradient: d c_i / d x_j = g(i*n+j)
        if (grad != nullptr)
        {
            for (unsigned int j = 0; j < n; ++j)
            {
                for (unsigned int i = 0; i < n; ++i)
                {
                    if (i == j)
                        grad[i * n + j] = -1.0;
                    else
                        grad[i * n + j] = 0.0;

                }

                grad[n * n + j] = 1.0;
            }
        }
    }

    static double oneSumConstraint(unsigned int n, const double *x,
                                   double *grad, void *data)
    {
        double val = -1.0;

        for (unsigned int i = 0; i < n; ++i)
            val += x[i];

        if (grad != nullptr)
            for (unsigned int i = 0; i < n; ++i)
                grad[i] = 1;

        return val;
    }

public:
    template<class Class, bool print = false>
    static double objFunctionWrapper(unsigned int n, const double* x, double* grad,
                                     void* o)
    {
        arma::vec df;
        arma::vec parV(const_cast<double*>(x), n, true);
        double value = static_cast<Class*>(o)->objFunction(parV, df);

        //Save gradient
        if (grad)
        {
            for (int i = 0; i < df.n_elem; ++i)
            {
                grad[i] = df[i];
            }
        }

        //Print gradient and value
        printOptimizationInfo<Class, print>(value, n, x, grad);

        return value;
    }

private:
    template<class Class, bool print>
    static void printOptimizationInfo(double value, unsigned int n, const double* x,
                                      double* grad)
    {
        if (print)
        {
            std::cout << "J(x) = " << value << " ";
            std::cout << "x = ";

            for (int i = 0; i < n; i++)
            {
                std::cout << x[i] << " ";
            }

            std::cout << std::endl;

            if (grad)
            {
                std::cout << "dJ/dx = ";

                for (int i = 0; i < n; i++)
                    std::cout << grad[i] << " ";

                std::cout << std::endl;
            }
        }
    }




};





}


#endif /* INCLUDE_RELE_IRL_UTILS_OPTIMIZATION_H_ */
