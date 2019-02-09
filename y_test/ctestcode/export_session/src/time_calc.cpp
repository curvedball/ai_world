
#include <common.h>
#include <time_calc.h>


//zb: 012345678
unsigned long get_seconds(string strSeconds)
{
    trim_string(strSeconds, " ,0", " ");
    return strtoul(strSeconds.c_str(), NULL, 0);
}


//zb: 012345678.11112222
unsigned long get_micro_seconds(string strDotMicroSeconds)
{
    vector<string> vec = split_string(strDotMicroSeconds, ".");
    unsigned  long v0 = get_seconds(vec[0]);
    unsigned  long v1 = get_seconds(vec[1]);
    return v0 * 1000000 + v1;
}




//===============================================================
//zb: 012345678 012345679
//
unsigned long get_delta(string str1, string str2)
{
    unsigned long v1 = get_seconds(str1);
    unsigned long v2 = get_seconds(str2);
    return v2 - v1;
}



//
unsigned long get_delta_seconds(unsigned long v1, unsigned long v2)
{
    return (v2 - v1) / 1000000;
}




unsigned long get_delta_milliseconds(unsigned long v1, unsigned long v2)
{
    return (v2 - v1) / 1000;
}



