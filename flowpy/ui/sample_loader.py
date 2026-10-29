"""Background loading of sample events.

Samples are opened from their headers only (instant); this loader then
reads their events in the background, one file at a time:
the sample being looked at first, then the others in list order. Anything
that needs events of samples not read yet calls wait_for(), which shows
progress (with Cancel) and moves those samples to the front.
"""

from __future__ import annotations
import threading
import time

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import QApplication, QProgressDialog

WORKERS = 1          # the sample being waited for gets the whole connection


class SampleLoader(QObject):
    """Signals (delivered in the GUI thread): loaded(sample) when a
    sample's events are ready (or failed: see sample.load_error);
    started(sample) when reading of a sample begins."""

    loaded = pyqtSignal(object)
    started = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._queue: list = []
        self._busy: set = set()
        self._cond = threading.Condition()
        self._stop = False
        self._threads = [threading.Thread(target=self._run, daemon=True,
                                          name=f"FlowPy loader {i}")
                         for i in range(WORKERS)]
        for t in self._threads:
            t.start()
        self.destroyed.connect(self.stop)

    def _emit(self, name: str, s):
        try:
            getattr(self, name).emit(s)
        except RuntimeError:          # window already closed
            self.stop()

    # ------------------------------------------------------------------

    def enqueue(self, samples, first=None):
        """Read *samples* (those not loaded yet) in this order; *first*
        goes to the front."""
        with self._cond:
            for s in samples:
                if not s.is_loaded and s not in self._queue \
                        and s not in self._busy and not s.load_error:
                    self._queue.append(s)
            if first is not None:
                self._front([first])
            self._cond.notify_all()

    def prioritize(self, samples):
        """Move *samples* to the front of the queue (adding them if
        needed), keeping their order."""
        with self._cond:
            self._front(list(samples))
            self._cond.notify_all()

    def _front(self, samples):
        todo = [s for s in samples if not s.is_loaded and not s.load_error
                and s not in self._busy]
        self._queue = todo + [s for s in self._queue if s not in todo]

    def discard(self, samples):
        """Stop waiting for *samples* (e.g. removed from the analysis)."""
        with self._cond:
            gone = {id(s) for s in samples}
            self._queue = [s for s in self._queue if id(s) not in gone]

    def clear(self):
        with self._cond:
            self._queue.clear()

    def pending(self) -> int:
        with self._cond:
            return len(self._queue) + len(self._busy)

    def is_busy(self, sample) -> bool:
        with self._cond:
            return sample in self._busy

    def stop(self):
        with self._cond:
            self._stop = True
            self._queue.clear()
            self._cond.notify_all()

    def _run(self):
        while True:
            with self._cond:
                while not self._queue and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                s = self._queue.pop(0)
                self._busy.add(s)
            self._emit("started", s)
            try:
                s.load()
            except Exception:
                pass                      # kept in s.load_error
            with self._cond:
                self._busy.discard(s)
            self._emit("loaded", s)

    # ------------------------------------------------------------------

    def wait_for(self, samples, parent=None, title="Reading samples…") -> bool:
        """Make sure *samples* have their events, with a progress dialog
        (and Cancel) if some are still to be read. Returns False if
        cancelled. Samples that can't be read are left out silently
        (their load_error tells why)."""
        samples = [s for s in samples if not s.is_loaded and not s.load_error]
        if not samples:
            return True
        self.prioritize(samples)
        n = len(samples)
        prog = QProgressDialog(title, "Cancel", 0, n, parent)
        prog.setWindowTitle("FlowPy")
        prog.setWindowModality(Qt.WindowModality.WindowModal)
        prog.setMinimumDuration(300)
        prog.setAutoClose(False)
        t0 = time.perf_counter()
        try:
            while True:
                done = sum(1 for s in samples if s.is_loaded or s.load_error)
                prog.setValue(done)
                left = ""
                el = time.perf_counter() - t0
                if 0 < done < n and el > 3:
                    sec = el / done * (n - done)
                    left = (f"  ·  about {sec / 60:.0f} min left" if sec >= 90
                            else f"  ·  about {sec:.0f} s left")
                prog.setLabelText(f"{title}\n{done} of {n} samples read"
                                  f"{left}\n(files are read only once)")
                if done >= n:
                    return True
                if prog.wasCanceled():
                    return False
                QApplication.processEvents()
                time.sleep(0.03)
        finally:
            prog.close()
