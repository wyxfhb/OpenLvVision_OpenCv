#include "pch.h"
#include <string>

void SetPathToCurrentDllDirectory() {
    HMODULE hModule = NULL;
    static int module = 0;

    GetModuleHandleExA
    (
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        (LPCSTR)&module,
        &hModule
    );

    if (hModule != NULL) 
    {
        char dllFilePath[MAX_PATH];

        if (GetModuleFileNameA(hModule, dllFilePath, MAX_PATH) != 0)
        {
            std::string fullPath(dllFilePath);
            size_t lastSlashIndex = fullPath.find_last_of("\\/");

            if (lastSlashIndex != std::string::npos) 
            {
                std::string dllDirectory = fullPath.substr(0, lastSlashIndex);
                SetDllDirectoryA(dllDirectory.c_str());
            }
        }
    }
}


BOOL APIENTRY DllMain( HMODULE hModule,
                       DWORD  ul_reason_for_call,
                       LPVOID lpReserved
                     )
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
        SetPathToCurrentDllDirectory();
        break;
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}

