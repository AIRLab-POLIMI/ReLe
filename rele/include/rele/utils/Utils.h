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

#ifndef INCLUDE_RELE_UTILS_UTILS_H_
#define INCLUDE_RELE_UTILS_UTILS_H_

#include <cmath>

class utils
{
public:
	static inline double normalizeAnglePositive(double angle)
	{
		return fmod(fmod(angle, 2.0 * M_PI) + 2.0 * M_PI, 2.0 * M_PI);
	}

	static inline double normalizeAngle(double angle)
	{
		angle = normalizeAnglePositive(angle);

		if (angle > M_PI)
		{
			angle -= 2.0 * M_PI;
		}

		return angle;
	}

	static inline double threshold(double value, double thresholdLow,
				double thresholdHigh)
	{
		return std::max(std::min(value, thresholdHigh), thresholdLow);
	}

	static inline double threshold(double value, double threshold)
	{
		return utils::threshold(value, threshold, -threshold);
	}

};

#endif /* INCLUDE_RELE_UTILS_UTILS_H_ */
