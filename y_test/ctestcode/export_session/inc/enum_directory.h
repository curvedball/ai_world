//
// Created by root on 19-1-19.
//

#ifndef EXTRACT_SESSION_ENUM_DIRECTORY_H
#define EXTRACT_SESSION_ENUM_DIRECTORY_H


#include <vector>
#include <iostream>
#include <string>
#include <vector>
#include <dirent.h>


using namespace std;


enum enum_type
{
    ENUM_DIR,
    ENUM_FILE
};


void enum_dir(string path, enum_type et, vector<string>& vector);


#endif //EXTRACT_SESSION_ENUM_DIRECTORY_H
