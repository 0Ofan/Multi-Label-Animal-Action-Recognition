"""Read Windows file allocation sizes; no writes or filesystem configuration changes."""
import ctypes
from ctypes import wintypes


class FileStandardInfo(ctypes.Structure):
    _fields_=[("AllocationSize",ctypes.c_longlong),("EndOfFile",ctypes.c_longlong),("NumberOfLinks",wintypes.DWORD),("DeletePending",ctypes.c_ubyte),("Directory",ctypes.c_ubyte)]


def allocation_bytes(paths):
    kernel=ctypes.WinDLL("kernel32",use_last_error=True)
    kernel.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    kernel.CreateFileW.restype=wintypes.HANDLE
    kernel.GetFileInformationByHandleEx.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD]
    kernel.GetFileInformationByHandleEx.restype=wintypes.BOOL
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    total=0;checked=0
    for path in paths:
        handle=kernel.CreateFileW(str(path),0x80,7,None,3,0,None)
        if handle in (None,ctypes.c_void_p(-1).value):raise OSError(ctypes.get_last_error(),"Cannot open allocation attributes",str(path))
        try:
            info=FileStandardInfo()
            if not kernel.GetFileInformationByHandleEx(handle,1,ctypes.byref(info),ctypes.sizeof(info)):raise OSError(ctypes.get_last_error(),"Cannot read file allocation",str(path))
            total+=info.AllocationSize;checked+=1
        finally:kernel.CloseHandle(handle)
    return dict(file_allocation_bytes=total,files_checked=checked,scope="Windows FILE_STANDARD_INFO.AllocationSize summed for regular cache files; directory/MFT overhead not included")
