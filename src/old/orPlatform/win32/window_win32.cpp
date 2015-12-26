#include "orPlatform/win32/window_win32.h"

#include <Windows.h>

struct HWND__;

void orPlatform::FocusWindow(void* winHandle) {
  HWND__* hwnd = (HWND__*)winHandle;
  ::SetForegroundWindow(hwnd);
  ::SetActiveWindow(hwnd);
  ::SetFocus(hwnd);
}