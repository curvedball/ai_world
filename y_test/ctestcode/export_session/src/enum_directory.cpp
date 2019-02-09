

#include <enum_directory.h>


void enum_dir(string path, enum_type et, vector<string>& vector)
{
    DIR* pDir = NULL;
    struct dirent* ent = NULL;
    pDir = opendir(path.c_str());
    if (NULL == pDir)
    {
        return;
    }

    while (NULL != (ent=readdir(pDir)))
    {
        if (ent->d_type == DT_DIR && string(ent->d_name) != "." && string(ent->d_name) != ".." && et == ENUM_DIR)
        {
            //cout << ent->d_name << endl;
            vector.push_back(string(ent->d_name));
        }
        else if(ent->d_type == DT_REG && et == ENUM_FILE)
        {
            //cout << ent->d_name << endl;
            vector.push_back(string(ent->d_name));
        }
        else
        {

        }
    }
    closedir(pDir);
    pDir = NULL;
    return;
}


