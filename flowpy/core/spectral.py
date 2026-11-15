"""Spectral detectors — find the full-spectrum detector channels of a
spectral cytometer (BD FACSDiscover / FACSymphony A5 SE, Cytek Aurora /
Northern Lights, Sony ID7000…) and put them in spectral order: laser by
laser (UV, violet, blue, yellow-green, red…), detector by detector.

Recognised names: "UV1 (375)-A", "V12 (625)-A" (BD, with the centre
wavelength), "UV1-A", "B3-A", "YG5-A", "R8-A" (Cytek).
"""

from __future__ import annotations
import re
from dataclasses import dataclass

# Laser letter, detector number, an optional centre wavelength in
# brackets, then the measurement kind. The laser group is non-greedy so
# that "YG5-A" splits as YG + 5 rather than Y + G5.
_DETECTOR = re.compile(
    r"^(?P<laser>[A-Za-z]{1,3}?)(?P<num>\d{1,2})"
    r"(?:\s*\((?P<nm>\d{3,4})\s*(?:nm)?\))?-(?P<kind>[AHW])$")
LASER_ORDER = ["UV", "V", "B", "YG", "Y", "G", "R", "IR"]
LASER_COLORS = {"UV": "#7c3aed", "V": "#4f46e5", "B": "#0ea5e9",
                "YG": "#84cc16", "Y": "#eab308", "G": "#22c55e",
                "R": "#ef4444", "IR": "#7f1d1d"}
#: Prefixes that look like a laser but are not. A conventional cytometer
#: numbers its channels FL1, FL2…, which matches the pattern above
#: perfectly and would otherwise be read as a laser called "FL".
NOT_LASERS = {"FL", "PMT", "APD"}

#: A real laser on a spectral instrument carries a whole detector array.
#: Imaging and scatter side-detectors come in ones and twos and share the
#: same naming, so a letter with only a few channels behind it is dropped.
MIN_PER_LASER = 4

#: Below this, it is not a spectral instrument. Guards against a
#: conventional panel whose channel names happen to fit the pattern.
MIN_DETECTORS = 12


@dataclass(frozen=True)
class Detector:
    channel: str
    laser: str
    number: int
    wavelength: int | None

    @property
    def short(self) -> str:
        return f"{self.laser}{self.number}"


def spectral_detectors(channels: list[str], kind: str = "A"
                       ) -> list[Detector]:
    """The spectral detectors among *channels* (area, height or width), in
    spectral order; [] if this doesn't look like a spectral instrument."""
    # Two passes: collect everything that parses, then decide whether the
    # collection as a whole looks like a spectral instrument. A per-channel
    # decision cannot tell the difference, since one detector name on a
    # conventional machine is indistinguishable from one on a spectral one.
    found = []
    for ch in channels:
        m = _DETECTOR.match(ch.strip())
        if m and m.group("kind") == kind:
            laser = m.group("laser").upper()
            if laser in NOT_LASERS:
                continue
            nm = m.group("nm")
            found.append(Detector(ch, laser, int(m.group("num")),
                                  int(nm) if nm else None))
    per = {}
    for d in found:
        per.setdefault(d.laser, []).append(d)
    # Drop the sparse groups — see MIN_PER_LASER.
    per = {k: v for k, v in per.items() if len(v) >= MIN_PER_LASER}
    # Spectral order, not alphabetical: a spectrum plotted UV -> IR is
    # readable, one plotted B, G, IR, R, UV, V, YG is not. Anything
    # unrecognised sorts to the end rather than being discarded.
    lasers = sorted(per, key=lambda l: (LASER_ORDER.index(l)
                                        if l in LASER_ORDER else 99, l))
    out = [d for l in lasers for d in sorted(per[l], key=lambda d: d.number)]
    return out if len(out) >= MIN_DETECTORS else []


def has_spectrum(channels: list[str]) -> bool:
    """Whether a spectrum plot makes sense for these channels.

    Area or height: some instruments export only one of the two, and
    either is a usable spectrum.
    """
    return bool(spectral_detectors(channels, "A") or
                spectral_detectors(channels, "H"))


def detector_kind(channels: list[str]) -> str:
    """'A' when area detectors exist, else 'H'."""
    return "A" if spectral_detectors(channels, "A") else "H"
