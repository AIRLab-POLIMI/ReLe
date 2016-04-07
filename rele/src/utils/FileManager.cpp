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

#include "rele/utils/FileManager.h"
#include <cstdlib>

using namespace std;

namespace ReLe
{

FileManager::FileManager(const string& environment,
                         const string& algorithm)
{
    outputDir = "/tmp/ReLe/" + environment + "/" + algorithm + "/";
}

FileManager::FileManager(const string& testName)
{
    outputDir = "/tmp/ReLe/" + testName + "/";
}

void FileManager::createDir()
{
    string createCommand = "mkdir -p " + outputDir;
    system(createCommand.c_str());
}

void FileManager::cleanDir()
{
    string cleanOldCommand = "rm -f " + outputDir + "*.log";
    system(cleanOldCommand.c_str());
}

string FileManager::addPath(const string& fileName)
{
    return outputDir + fileName;
}

string FileManager::addPath(const string& prefix, const string& fileName)
{
    return outputDir + prefix + "_" + fileName ;
}

}
