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

#ifndef INCLUDE_RELE_IRL_UTILS_IRLGRADTYPE_H_
#define INCLUDE_RELE_IRL_UTILS_IRLGRADTYPE_H_

#include <map>
#include <stdexcept>

namespace ReLe
{

enum class IrlGrad
{
    REINFORCE, REINFORCE_BASELINE,
    GPOMDP, GPOMDP_BASELINE,
    ENAC, ENAC_BASELINE,
    NATURAL_REINFORCE, NATURAL_REINFORCE_BASELINE,
    NATURAL_GPOMDP, NATURAL_GPOMDP_BASELINE
};

enum class IrlEpGrad
{
    PGPE, PGPE_BASELINE
};

enum class IrlHess
{
    REINFORCE, REINFORCE_BASELINE,
    REINFORCE_BASELINE_TRACE, REINFORCE_BASELINE_TRACE_DIAG,
    GPOMDP, GPOMDP_BASELINE
};

enum class IrlEpHess
{
    PGPE, PGPE_BASELINE
};

class IrlGradUtils
{
public:
    static bool isValid(const std::string& type);
    static IrlGrad fromString(const std::string& type);
    static std::string toString(IrlGrad type);
    static std::string getOptions();

private:
    static std::map<std::string, IrlGrad> initGradients();

private:
    static std::map<std::string, IrlGrad> gradients;


};

class IrlEpGradUtils
{
public:
    static bool isValid(const std::string& type);
    static IrlEpGrad fromString(const std::string& type);
    static std::string toString(IrlEpGrad type);
    static std::string getOptions();

private:
    static std::map<std::string, IrlEpGrad> initGradients();

private:
    static std::map<std::string, IrlEpGrad> gradients;


};

class IrlHessUtils
{
public:
    static bool isValid(const std::string& type);
    static IrlHess fromString(const std::string& type);
    static std::string toString(IrlHess type);
    static std::string getOptions();


private:
    static std::map<std::string, IrlHess> initHessians();

private:
    static std::map<std::string, IrlHess> hessians;


};


class IrlEpHessUtils
{
public:
    static bool isValid(const std::string& type);
    static IrlEpHess fromString(const std::string& type);
    static std::string toString(IrlEpHess type);
    static std::string getOptions();


private:
    static std::map<std::string, IrlEpHess> initHessians();

private:
    static std::map<std::string, IrlEpHess> hessians;


};


inline std::ostream& operator<< (std::ostream& stream, IrlGrad grad)
{
    stream << IrlGradUtils::toString(grad);
    return stream;
}

inline std::ostream& operator<< (std::ostream& stream, IrlEpGrad grad)
{
    stream << IrlEpGradUtils::toString(grad);
    return stream;
}


inline std::ostream& operator<< (std::ostream& stream, IrlHess hess)
{
    stream << IrlHessUtils::toString(hess);
    return stream;
}

inline std::ostream& operator<< (std::ostream& stream, IrlEpHess hess)
{
    stream << IrlEpHessUtils::toString(hess);
    return stream;
}


}



#endif /* INCLUDE_RELE_IRL_UTILS_IRLGRADTYPE_H_ */
