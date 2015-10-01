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

#ifndef INCLUDE_ALGORITHMS_TD_Q_LEARNING_H_
#define INCLUDE_ALGORITHMS_TD_Q_LEARNING_H_

#include "TD.h"

namespace ReLe
{

/**
 * http://webdocs.cs.ualberta.ca/~sutton/book/ebook/node65.html
 */
class Q_Learning: public FiniteTD
{
public:
    Q_Learning(ActionValuePolicy<FiniteState>& policy);
    virtual void initEpisode(const FiniteState& state, FiniteAction& action) override;
    virtual void sampleAction(const FiniteState& state, FiniteAction& action) override;
    virtual void step(const Reward& reward, const FiniteState& nextState,
                      FiniteAction& action) override;
    virtual void endEpisode(const Reward& reward) override;


    virtual ~Q_Learning();

};

}



#endif /* INCLUDE_ALGORITHMS_TD_Q_LEARNING_H_ */
