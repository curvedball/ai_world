



#ifndef EXTRACT_SESSION_COMMON_H
#define EXTRACT_SESSION_COMMON_H


#include <stdio.h>
#include <vector>
#include <iostream>
#include <string>
#include <vector>
#include <dirent.h>
#include <string>
#include <stdlib.h>
#include <string.h>
#include <fstream>



using namespace std;


string format_string(const char* fmt, ...);


//
string get_program_absolute_path(const char* program);

//
string get_dir(const string& strPath);

//
void mkdirs(const char *dir);



//
void split_string(const string& str, const string& delim, vector<string>& vec);
//
vector<string> split_string(const string& str, const string& delim);

//
//vector<string> split_string_strictly(string str, const string& pattern);


//
string& trim_string(string &s, const string& strLeftPattern, const string& strRightPattern);


//
unsigned long get_ul(string str);



#endif //EXTRACT_SESSION_COMMON_H
