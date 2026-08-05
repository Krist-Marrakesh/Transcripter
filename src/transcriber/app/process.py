"""Asking the operating system about another process.

Three questions are put to a running copy of ourselves: is it alive, when was it
started, and can its window be raised. Every system answers them its own way and
the ways have nothing in common — `ps` and AppleScript on macOS, kernel calls
through ctypes on Windows. Keeping both inside `single` would bury the decision
that module exists to make — open a new window or raise the open one — under the
details of two platforms.

One trap is worth naming. The null signal that asks "are you alive" on POSIX is
not a question on Windows: CPython implements `os.kill` through
`TerminateProcess` for every signal but the two console ones, so the check would
kill the very copy it was looking for. Hence a separate branch rather than shared
code with a footnote.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys

if sys.platform == "win32":
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _user32 = ctypes.WinDLL("user32", use_last_error=True)

    _VISIT = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    _FILETIME = ctypes.POINTER(wintypes.FILETIME)

    # Prototypes are spelled out rather than left to ctypes' defaults: without an
    # explicit `restype` a returned value is taken for a C int, and a 64-bit
    # handle loses its upper half. The call then succeeds against a handle that
    # points nowhere, which is far harder to see than a plain failure.
    _kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    _kernel32.CloseHandle.restype = wintypes.BOOL
    _kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    _kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    _kernel32.GetProcessTimes.argtypes = (wintypes.HANDLE, *(_FILETIME,) * 4)
    _kernel32.GetProcessTimes.restype = wintypes.BOOL

    _user32.EnumWindows.argtypes = (_VISIT, wintypes.LPARAM)
    _user32.EnumWindows.restype = wintypes.BOOL
    _user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
    _user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    _user32.GetWindow.argtypes = (wintypes.HWND, wintypes.UINT)
    _user32.GetWindow.restype = wintypes.HWND
    _user32.IsWindowVisible.argtypes = (wintypes.HWND,)
    _user32.IsWindowVisible.restype = wintypes.BOOL
    _user32.ShowWindow.argtypes = (wintypes.HWND, ctypes.c_int)
    _user32.ShowWindow.restype = wintypes.BOOL
    _user32.SetForegroundWindow.argtypes = (wintypes.HWND,)
    _user32.SetForegroundWindow.restype = wintypes.BOOL

    # The narrowest right that still answers both questions below. Asking for
    # more would make the check fail on processes we are merely looking at.
    _QUERY = 0x1000  # PROCESS_QUERY_LIMITED_INFORMATION
    _RUNNING = 259  # STILL_ACTIVE
    _RESTORE = 9  # SW_RESTORE
    _OWNER = 4  # GW_OWNER

    def alive(pid: int) -> bool:
        """Whether a process with this number exists.

        An open handle alone proves nothing: the record survives until the last
        handle to it is closed, so a finished process still opens. The exit code
        tells the two apart — with one known corner, a process that really did
        exit with code 259 looks alive. It costs a refused second window, not a
        killed first one, so the trade is worth it.
        """
        handle = _kernel32.OpenProcess(_QUERY, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            if not _kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == _RUNNING
        finally:
            _kernel32.CloseHandle(handle)

    def started_at(pid: int) -> str:
        """When the process started, in system ticks. Empty string — no answer."""
        handle = _kernel32.OpenProcess(_QUERY, False, pid)
        if not handle:
            return ""
        try:
            created = wintypes.FILETIME()
            # The other three times are of no use to us, but the call fills all
            # four and refuses null pointers.
            spare = (wintypes.FILETIME * 3)()
            filled = _kernel32.GetProcessTimes(
                handle,
                ctypes.byref(created),
                ctypes.byref(spare[0]),
                ctypes.byref(spare[1]),
                ctypes.byref(spare[2]),
            )
            if not filled:
                return ""
            return str((created.dwHighDateTime << 32) | created.dwLowDateTime)
        finally:
            _kernel32.CloseHandle(handle)

    def focus(pid: int) -> bool:
        """Raises the window belonging to the process.

        There is no way to ask for "the window of process N" — only to walk every
        top-level window and check who holds it.
        """
        found: list[int] = []

        @_VISIT
        def visit(window: int, _: int) -> bool:
            holder = wintypes.DWORD()
            _user32.GetWindowThreadProcessId(window, ctypes.byref(holder))
            if holder.value != pid or not _user32.IsWindowVisible(window):
                return True
            # Dialogs and tooltips belong to a main window and would come forward
            # instead of it; the one we want owns itself.
            if _user32.GetWindow(window, _OWNER):
                return True
            found.append(window)
            return False

        _user32.EnumWindows(visit, 0)
        if not found:
            return False
        # Without this a minimised window is activated while staying minimised —
        # from the outside the second launch then does nothing at all.
        _user32.ShowWindow(found[0], _RESTORE)
        return bool(_user32.SetForegroundWindow(found[0]))

else:

    def alive(pid: int) -> bool:
        """Whether a process with this number exists.

        The null signal sends nothing — it only asks whether there is someone to
        send to.
        """
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            # It exists but belongs to another user — so it is not ours.
            return False
        return True

    def started_at(pid: int) -> str:
        """When the process started. Empty string — no answer."""
        done = subprocess.run(
            ["ps", "-p", str(pid), "-o", "lstart="], capture_output=True, text=True, check=False
        )
        return done.stdout.strip()

    def focus(pid: int) -> bool:
        """Raises the window of a running copy to the front.

        Through System Events, because our bundle is script-based: the process
        has no application identifier of its own to take hold of otherwise.
        """
        if sys.platform != "darwin":
            return False
        script = (
            'tell application "System Events" to set frontmost of '
            f"(first process whose unix id is {pid}) to true"
        )
        done = subprocess.run(["osascript", "-e", script], capture_output=True, check=False)
        return done.returncode == 0
