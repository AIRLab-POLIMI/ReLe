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

#ifndef INCLUDE_RELE_UTILS_FILEMANAGER_H_
#define INCLUDE_RELE_UTILS_FILEMANAGER_H_

#include <string>

namespace ReLe
{
/*!
 * This class has some useful functions to manage I/O of data files.
 */
class FileManager
{
public:
    /*!
     * Constructor.
     * It creates the path where the file will be saved or loaded.
     * \param environment name of the environment in which the algorithm is executed
     * \param algorithm name of the algorithm to be performed
     */
    FileManager(const std::string& environment, const std::string& algorithm);

    /*!
     * Constructor.
     * It creates the path where the file will be saved or loaded.
     * \param testName name of the test to be performed
     */
    FileManager(const std::string& testName);

    /*!
     * Creates the directory at the path given by the constructor.
     */
    void createDir();

    /*!
     * Cleans the directory at the path given by the constructor parameters
     */
    void cleanDir();

    /*!
     * Add the name of the file to the path created in the constructor.
     * \param fileName name of the file
     * \return the path string
     */
    std::string addPath(const std::string& fileName);

    /*!
     * Add the name of the file to the path created in the constructor.
     * \param prefix the prefix before the name of the file which is separated by '_'
     * \param fileName name of the file
     * \return the path string
     */
    std::string addPath(const std::string& prefix, const std::string& fileName);

private:
    std::string outputDir;

};

}

#endif /* INCLUDE_RELE_UTILS_FILEMANAGER_H_ */
