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

#include <nlopt.hpp>
#include <iostream>

double f(unsigned int n, const double* x, double* grad, void*)
{
	static int it = 0;

	double a = x[0];
	double b = x[1];
	double c = x[2];

	grad[0] = 2*a*b + 2*a*c + b*b + c*c + 1;
	grad[1] = a*a + c*c + 2*b*a + 2*b*c + 1;
	grad[2] = a*a + b*b + 2*c*a + 2*c*b + 1;

	std::cout << it++ << std::endl;

	return a*a*b + a*a*c + b*b*a + b*b*c + c*c*a + c*c*b + a + b + c;

}

double g(unsigned int n, const double* x, double* grad, void*)
{
	static int it = 0;

	double a = x[0];
	double b = x[1];
	double c = x[2];

	grad[0] = 2*(a - 3)*std::exp((a - 3)*(a - 3) - b*b + c*c) - 2*(a - 3)*std::exp(-(a - 3)*(a - 3) + b*b - c*c + 100);
	grad[1] = -2*b*std::exp((a - 3)*(a - 3) - b*b + c*c) + 2*b*std::exp(-(a - 3)*(a - 3) + b*b - c*c + 100);
	grad[2] = 2*c*std::exp((a - 3)*(a - 3) - b*b + c*c) - 2*c*std::exp(-(a - 3)*(a - 3) + b*b - c*c + 100);

	std::cout << it++ << std::endl;

	return std::exp((a - 3)*(a - 3) - b*b + c*c) + std::exp(-(a - 3)*(a - 3) + b*b - c*c + 100);

}

int main()
{
	nlopt::opt optimizator;
	int size = 3;
	optimizator = nlopt::opt(nlopt::algorithm::LD_LBFGS, size);
	//optimizator = nlopt::opt(nlopt::algorithm::LD_MMA, size);
	//optimizator.set_min_objective(f, NULL);
	optimizator.set_min_objective(g, NULL);
	optimizator.set_xtol_rel(1e-8);
	optimizator.set_ftol_rel(1e-12);
	optimizator.set_maxeval(5);

	std::vector<double> x(size, 1);
	double J;
	optimizator.optimize(x, J);

	std::cout << "x = " << x[0] << ", " << x[1] << ", " << x[2] << std::endl;
	std::cout << "J = " << J;

}
