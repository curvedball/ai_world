//
// Created by root on 19-1-19.
//

#ifndef EXTRACT_SESSION_TIME_CALC_H
#define EXTRACT_SESSION_TIME_CALC_H


#include <stdio.h>
#include <vector>
#include <iostream>
#include <string>
#include <vector>
#include <dirent.h>
#include <string>


using namespace std;


//
unsigned long get_seconds(string strSeconds);


//
unsigned long get_micro_seconds(string strDotMicroSeconds);



//
unsigned long get_delta(string str1, string str2);

//
unsigned long get_delta_seconds(unsigned long v1, unsigned long v2);

//
unsigned long get_delta_milliseconds(unsigned long v1, unsigned long v2);


#endif //EXTRACT_SESSION_TIME_CALC_H
