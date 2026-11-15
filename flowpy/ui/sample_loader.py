"""Background loading of sample events.

Samples are opened from their headers only (instant); this loader then
reads their events in the background, one file at a time:
the sample being looked at first, then the others in list order. Anything
that needs events of samples not read yet calls wait_for(), which shows
progress (with Cancel) and moves those samples to the front.
"""

# ---------------------------------------------------------------------------
# HOW THE BACKGROUND LOADER WORKS
# ---------------------------------------------------------------------------
# Opening an FCS file reads only its header, which is instant. The
# events are read here instead, on a worker thread, so a folder of large
# files opens immediately and fills in behind the user.
#
#   _queue    samples still to read, in priority order
#   _busy     samples being read right now
#   _cond     one Condition guarding both, and used to wake the worker
#
# The worker loops in _run(): wait on the condition, take the front of
# the queue, read it, emit "loaded". Everything that touches the queue
# does so under `with self._cond`, because the GUI thread adds to it
# while the worker takes from it.
#
# wait_for() is the bridge back to synchronous code: when something
# needs events that are not read yet, it pushes those samples to the
# front and spins a progress dialog until they arrive.
#
# Signals carry the work back to the GUI thread. Qt queues them across
# the thread boundary automatically, which is why the worker may emit
# directly rather than marshalling by hand.
# ---------------------------------------------------------------------------

from __future__ import annotations
import threading
import time

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import QApplication, QProgressDialog

#: One worker on purpose. These files often sit on a network share,
#: where several parallel reads are slower than one — and with a single
#: worker the sample being waited for gets the whole connection instead
#: of competing with the ones nobody is looking at.
WORKERS = 1


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
        """Emit a signal, tolerating the window having gone.

        The worker outlives the window during shutdown, and emitting
        from a deleted QObject raises RuntimeError. Catching it and
        stopping is the orderly end; letting it propagate would kill the
        thread with a traceback on every quit.
        """
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
            # Wake the worker: it is asleep on the condition whenever the
            # queue is empty, and would not notice the new work otherwise.
            self._cond.notify_all()

    def prioritize(self, samples):
        """Move *samples* to the front of the queue (adding them if
        needed), keeping their order."""
        with self._cond:
            self._front(list(samples))
            self._cond.notify_all()

    def _front(self, samples):
        """Move *samples* to the head of the queue, keeping their order.

        Must be called with the condition held. Already-loaded, failed
        and in-progress samples are filtered out first — re-queueing one
        that is being read would read it twice.
        """
        todo = [s for s in samples if not s.is_loaded and not s.load_error
                and s not in self._busy]
        # Rebuilt rather than reordered, so a sample already further back
        # in the queue is moved rather than duplicated.
        self._queue = todo + [s for s in self._queue if s not in todo]

    def discard(self, samples):
        """Stop waiting for *samples* (e.g. removed from the analysis)."""
        with self._cond:
            # Compared by identity, not equality: two samples may be
            # equal by value while being different open files.
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
        """The worker loop. One of these per thread in WORKERS."""
        while True:
            with self._cond:
                # `while`, not `if`: wait() can return without the
                # condition being true (spurious wakeups, or another
                # worker taking the item first), so the test is repeated.
                while not self._queue and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                s = self._queue.pop(0)
                self._busy.add(s)
            # Read outside the lock — this is the slow part, and holding
            # the condition here would block the GUI thread from queueing
            # anything for as long as the file takes.
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
        # Nothing to wait for is the common case — say yes without
        # flashing a dialog.
        samples = [s for s in samples if not s.is_loaded and not s.load_error]
        if not samples:
            return True
        self.prioritize(samples)
        n = len(samples)
        prog = QProgressDialog(title, "Cancel", 0, n, parent)
        prog.setWindowTitle("FlowPy")
        prog.setWindowModality(Qt.WindowModality.WindowModal)
        # Nothing appears for the first 300 ms, so a quick read does not
        # flash a dialog. autoClose off because the loop below closes it
        # in the finally, and letting Qt close it too causes a flicker.
        prog.setMinimumDuration(300)
        prog.setAutoClose(False)
        t0 = time.perf_counter()
        try:
            while True:
                done = sum(1 for s in samples if s.is_loaded or s.load_error)
                prog.setValue(done)
                left = ""
                el = time.perf_counter() - t0
                # An estimate only after 3 seconds and one finished
                # file: before that the rate is one sample over a very
                # short time, and the figure jumps around absurdly.
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
                # Spin rather than block: the reading happens on the
                # worker, and this loop exists only to keep the dialog
                # painting and responsive to Cancel. The 30 ms sleep
                # keeps that at about 30 fps without busy-waiting.
                QApplication.processEvents()
                time.sleep(0.03)
        finally:
            prog.close()
