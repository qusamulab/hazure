"""The selected cut-off and the objective curve behind it."""

from __future__ import annotations

from dataclasses import dataclass

from hazure.thresholds import FixedThreshold


@dataclass(frozen=True, slots=True)
class Calibration:
    """Where the line was drawn, what that achieved, and what else was considered.

    Attributes
    ----------
    cut_off : float
        The chosen cut-off. Scores strictly above it are anomalous.
    score : float
        What :attr:`objective` came to at :attr:`cut_off`.
    objective : str
        What was being optimised, as text — ``"event f1"``, ``"alerts per 7d"``.
        Worth carrying, because a bare number two functions could both have
        produced is not self-describing.
    curve : tuple of tuple
        Every ``(cut_off, score)`` pair examined, in increasing order of cut-off.
        This is the part worth looking at: a peak the width of one candidate is a
        peak fitted to noise, and no summary statistic will tell you that.

    See Also
    --------
    tune_threshold : Choose a cut-off against labelled history.
    budget_threshold : Choose a cut-off against an alert budget.

    Examples
    --------
    >>> calibration = Calibration(2.5, 0.8, "event f1", ((1.0, 0.5), (2.5, 0.8)))
    >>> calibration.threshold
    FixedThreshold(high=2.5)
    """

    cut_off: float
    score: float
    objective: str
    curve: tuple[tuple[float, float], ...]

    @property
    def threshold(self) -> FixedThreshold:
        """The cut-off as a threshold, ready to pair with a scorer.

        Returns
        -------
        FixedThreshold
            A one-sided fence at :attr:`cut_off`. Pair it with the scorer the
            calibration was computed from, through
            :class:`~hazure.Detector`.
        """
        return FixedThreshold(high=self.cut_off)

    def __repr__(self) -> str:
        return (
            f"Calibration(cut_off={self.cut_off:.6g}, {self.objective}="
            f"{self.score:.6g}, over {len(self.curve)} candidates)"
        )
