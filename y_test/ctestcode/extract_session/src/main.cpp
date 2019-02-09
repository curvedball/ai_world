



#include <common.h>
#include <enum_directory.h>
#include <ini_parser.h>
#include <time_calc.h>


//
#define TARGET_DATA_LEN 784



//
string strProgramPath;
string strProgramDir;
string strProto;
string strIniFilePath;
string strPcapDir;
string strOutputDir;


//zb: To increase performance
string strCurSrcDirPath;
string strCurDstDirPath;





//zb: "aaa_UDP_10.138.20.12_10000_10.138.20.140_5000_15673465.111222_15673469.111333_37.txt";
int ProcessFile(string strFileName, int class_value)
{
    string strSrcFilePath = strCurSrcDirPath + "/" + strFileName;
    string strDstFilePath = strCurDstDirPath + "/" + strFileName;

    //
    vector<string> vec;
    split_string(strFileName, "_", vec);
    unsigned short flowDelta = get_delta(vec[6], vec[7]);
    //printf("time1: %s time2: %s diff: %u  [%s]\n", vec[6].c_str(), vec[7].c_str(), flowDelta, strFileName.c_str());
    if (flowDelta < 100)
    {
        //return -1;
    }

    //=====================================================================
    char data_buf[TARGET_DATA_LEN];
    memset(data_buf, 0, TARGET_DATA_LEN);
    unsigned short* pDataStart = (unsigned short*)data_buf;
    unsigned short* pData = pDataStart;
    int win_count = 0; //20 windows
    int pkt_count = 0; //10 packets in each window(6 packets in the final window)

    //
    unsigned long t_start;
    unsigned long t1;
    unsigned long t2;
    unsigned short span;
    unsigned short delta;
    unsigned short pkt_len;
    bool flag = false;

    //
    fstream f(strSrcFilePath.c_str());
    vector<string>  vecField;
    string line;
    while(getline(f, line)) //会自动把\n换行符去掉
    {
        vector<string>().swap(vecField);
        split_string(line, " ,\t", vecField);

        t2 = get_micro_seconds(vecField[1]);
        pkt_len = (unsigned short)get_ul(vecField[2]);
        if (!flag)
        {
            t_start = t2;
            t1 = t2;
            flag = true;
        }
        span = (unsigned short)get_delta_seconds(t_start, t2);
        delta = (unsigned short)get_delta_milliseconds(t1, t2);
        printf("span: [%u]  delta: %u  pkt_len: %u\n", span, delta, pkt_len);


        //================================================================
        //100second: 20 periods(windows)
        if (span >= (win_count + 1) * 5) //zb: 5 seconds/window (10 packets in this window).
        {
            //win_count++;
            win_count = span / 5;
            pkt_count = 0;
            pData = pDataStart + 20 * win_count; //10 packets, 2 unsigned short values for each packet.

        }

        //==================================================================
        if (win_count < 19)
        {
            if (pkt_count < 10)
            {
                *pData++ = delta;
                *pData++ = pkt_len;
                pkt_count++;
            }
        }
        else if (win_count == 19)
        {
            if (pkt_count < 6) //The final window: 6 packets.
            {
                *pData++ = delta;
                *pData++ = pkt_len;
                pkt_count++;
            }
        }
        else
        {
            break;
        }

        //===================================================================
        t1 = t2;
    }

    //zb: 196 packets if there are enough packets.
    printf("data_len: %d\n", (unsigned int)((pData - pDataStart) * 2));


    //
    ofstream ofs(strDstFilePath.c_str(), ofstream::binary); //二进制格式
    if (ofs.is_open())
    {
        ofs.write(data_buf, TARGET_DATA_LEN);
        ofs.close();
    }

    return 0;
}



int ProcessSubClass(string path, string dir_name, int class_value)
{
    string cur_path = path + "/" + dir_name;
    string strInfo = format_string("[%s]", dir_name.c_str());
    cout << strInfo << endl;

    //
    strCurSrcDirPath = cur_path;
    strCurDstDirPath = format_string("%s/%d", strOutputDir.c_str(), class_value);

    //
    vector<string> v;
    enum_dir(cur_path, ENUM_FILE, v);

    vector<string>::iterator it = v.begin();
    while (it != v.end())
    {
        cout << *it << endl;
        ProcessFile(*it, class_value);
        it++;
    }

    return 0;
}




int ProcessTrafficClass(string path, string dir_name, int class_value)
{
    string cur_path = path + "/" + dir_name;
    string strInfo = format_string("TrafficClass: [%d] path: %s", class_value, cur_path.c_str());
    cout << strInfo << endl;

    //
    string strDstDir = format_string("%s/%d", strOutputDir.c_str(), class_value);
    mkdirs(strDstDir.c_str());

    //
    vector<string> v;
    enum_dir(cur_path, ENUM_DIR, v);

    vector<string>::iterator it = v.begin();
    while (it != v.end())
    {
        ProcessSubClass(cur_path, *it, class_value);
        it++;
    }
    cout << endl;

    return 0;
}




int main(int argc, char** argv)
{
    if (argc == 1)
    {
        strProto = "UDP";
    }
    else if (argc == 2)
    {
        string strArg1 = string(argv[1]);
        if (strArg1 != "UDP" &&  strArg1 != "TCP")
        {
            cout << "The parameter must be UDP or TCP!" << endl;
            return -1;
        }
        strProto = strArg1;
    }
    else
    {
        cout << "Usage: ./program proto   (proto must be UDP or TCP)" << endl;
        return -1;
    }

    //
    strProgramPath = get_program_absolute_path(argv[0]);
    //cout << strProgramPath << endl;
    strProgramDir = get_dir(strProgramPath);
    //cout << strProgramDir << endl;
    strIniFilePath = strProgramDir + "/extract_session.ini";
    //cout << strIniFilePath << endl;

    //
    //cout << strProto << endl;
    CMyINI myINI;
    myINI.ReadINI(strIniFilePath);
    //myINI.Travel();
    strPcapDir = myINI.GetValue(strProto, "pcap_dir");
    //cout << strPcapDir << endl;
    strOutputDir = myINI.GetValue(strProto, "output_dir");
    //cout << strOutputDir << endl;
    mkdirs(strOutputDir.c_str());


    //=======================================================
    string cur_path = strPcapDir;
    vector<string> v;
    enum_dir(cur_path, ENUM_DIR, v);

    //
    int class_value = 0;
    vector<string>::iterator it = v.begin();
    while (it != v.end())
    {
        ProcessTrafficClass(cur_path, *it, class_value);
        it++;
        class_value++;
    }

}









