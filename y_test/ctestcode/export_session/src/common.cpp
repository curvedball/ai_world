

#include <common.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>



string format_string(const char* fmt, ...)
{
    char buf[1024];
    va_list va;
    va_start(va, fmt);
    vsprintf(buf, fmt, va);
    va_end(va);
    return string(buf);
}


string get_program_absolute_path(const char* program)
{
    char  resolved_path[256];
    realpath(program, resolved_path);
    return string(resolved_path);  //return string is OK(will be constructed), but performance should be considered!
}


string get_dir(const string& strPath)
{
    string::size_type pos = strPath.find_last_of('/', strPath.npos);
    //cout << pos << endl;
    if (pos == 0)
    {
        return string("/");
    }
    return strPath.substr(0, pos);
}



void mkdirs(const char *dir)
{
    char tmp[1024];
    char *p;

    if (strlen(dir) == 0 || dir == NULL)
    {
        printf("strlen(dir) is 0 or dir is NULL./n");
        return;
    }

    memset(tmp, 0, sizeof(tmp));
    strncpy(tmp, dir, strlen(dir));
    if (tmp[0] == '/')
    {
        p = strchr(tmp + 1, '/');
    }
    else
    {
        p = strchr(tmp, '/');
    }

    if (p)
    {
        *p = '\0';
        mkdir(tmp, 0777);
        chdir(tmp);
    }
    else
    {
        mkdir(tmp, 0777);
        chdir(tmp);
        return;
    }

    mkdirs(p + 1);
}



void split_string(const string& str, const string& delim, vector<string>& vec) //zb: High performance version
{
    if("" == str)
    {
        return;
    }

    //先将要切割的字符串从string类型转换为char*类型
    char * strs = new char[str.length() + 1] ; //不要忘了
    strcpy(strs, str.c_str());

    char * d = new char[delim.length() + 1];
    strcpy(d, delim.c_str());

    char *p = strtok(strs, d);
    while(p)
    {
        string s = p; //分割得到的字符串转换为string类型
        vec.push_back(s); //存入结果数组
        p = strtok(NULL, d);
    }
}



vector<string> split_string(const string& str, const string& delim) //zb: Low performance version
{
    vector<string> res;
    if("" == str)
    {
        return res;
    }

    //先将要切割的字符串从string类型转换为char*类型
    char * strs = new char[str.length() + 1] ; //不要忘了
    strcpy(strs, str.c_str());

    char * d = new char[delim.length() + 1];
    strcpy(d, delim.c_str());

    char *p = strtok(strs, d);
    while(p)
    {
        string s = p; //分割得到的字符串转换为string类型
        res.push_back(s); //存入结果数组
        p = strtok(NULL, d);
    }
    return res;
}


/*
vector<string> split_string_strictly(string str, const string& pattern)
{
    string::size_type pos;
    vector<string> result;
    str += pattern;//扩展字符串以方便操作
    int size=str.size();

    for(int i=0; i<size; i++)
    {
        pos=str.find(pattern,i);
        if(pos<(unsigned int)size)
        {
            std::string s=str.substr(i,pos-i);
            result.push_back(s);
            i=pos+pattern.size()-1;
        }
    }
    return result;
}
*/


string& trim_string(string &s, const string& strLeftPattern, const string& strRightPattern)
{
    if (s.empty())
    {
        return s;
    }
    s.erase(0,s.find_first_not_of(strLeftPattern));
    s.erase(s.find_last_not_of(strRightPattern) + 1);
    return s;
}



unsigned long get_ul(string str)
{
    trim_string(str, " ,0", " ");
    return strtoul(str.c_str(), NULL, 0);
}



//6video_http, 100unknown
string get_classname_from_dirname(string str)
{
    string s = trim_string(str, " ,0", " ");
    unsigned int i;
    for (i = 0; i < s.length(); i++)
    {
        if (s[i] < '0' || s[i] > '9')
        {
            break;
        }
    }

    if (i == 0)
    {
        cout << "Wrong dirname: " << str << endl;
        exit(-1);
    }
    return s.substr(0, i);
}







