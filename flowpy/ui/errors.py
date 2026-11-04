"""One way of showing an error, with the traceback kept.

FlowPy used to put ``str(exc)`` in a message box and throw the rest away.
That is fine for the errors it raises itself, which are sentences written
for the person reading them. It is useless for the ones it does not: a
dialog saying only

    index -5860 is out of bounds for axis 0 with size 4097

names no file, no line and no operation, and the only way to find out
where it came from was to relaunch the app from a terminal — which is not
something to ask of a colleague on another machine, and not possible at
all once the app has been handed over as a download.

So the message stays as the headline and the traceback goes underneath,
behind "Show Details". Copy-and-pasteable, and it costs nothing when
nothing goes wrong.
"""

from __future__ import annotations

import traceback

from PyQt6.QtWidgets import QMessageBox


def _detail(exc: BaseException | None) -> str:
    """The traceback for *exc*, or whatever is currently being handled."""
    if exc is not None and getattr(exc, "__traceback__", None) is not None:
        return "".join(traceback.format_exception(
            type(exc), exc, exc.__traceback__))
    text = traceback.format_exc()
    # format_exc() outside an except block is the string "NoneType: None",
    # which would be a worse thing to show than nothing.
    return "" if text.startswith("NoneType") else text


def show_error(parent, title: str, exc: BaseException | str,
               hint: str = "") -> None:
    """An error box whose details hold the traceback.

    *exc* may be the exception or a message. *hint* adds a line of advice
    above the technical text — what to try, rather than what went wrong.
    """
    message = str(exc) or type(exc).__name__
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Critical)
    box.setWindowTitle(title)
    box.setText(message)
    if hint:
        box.setInformativeText(hint)
    detail = _detail(exc if isinstance(exc, BaseException) else None)
    if detail:
        box.setDetailedText(
            f"{title}\n\n{detail}\n"
            "Copy this whole box when reporting the problem — the message "
            "on its own does not say where it happened.")
    box.setStandardButtons(QMessageBox.StandardButton.Ok)
    box.exec()
