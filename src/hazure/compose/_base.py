"""Output dispatch and validation shared by composite components."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from hazure._core.component import Component

if TYPE_CHECKING:
    from collections.abc import Sequence

    from hazure._core.component import OutputKind


# Reserved node name for the data supplied to the graph.
SOURCE = "input"


_VERBS: dict[str, tuple[str, ...]] = {
    "score": ("score",),
    "series": ("transform",),
    "labels": ("detect", "apply"),
}


_OUTPUT_DESCRIPTION: dict[str, str] = {
    "score": "continuous scores",
    "series": "a transformed series",
    "labels": "binary labels",
}


class _Composite(Component):
    """Shared behaviour for structures whose output type depends on their parts.

    A composite answers to whichever verb its terminal component answers to.
    Calling the wrong one raises rather than quietly returning the wrong kind of
    thing, and says which verb to use instead.
    """

    # A composite takes whatever it is given and lets each part decide: a
    # univariate step fans out over columns on its own, and a multivariate step
    # needs them all. Fanning out at the composite level would deny the second
    # kind the columns it needs.
    multivariate: ClassVar[bool] = True

    @property
    def terminal(self) -> Component:
        """The component whose output is this structure's output."""
        raise NotImplementedError

    @property
    def is_trainable(self) -> bool:
        """True when any part has something to learn."""
        return any(part.is_trainable for part in self._parts().values())

    @property
    def output_kind(self) -> OutputKind:
        """What the terminal component emits.

        What matters is the *kind of thing* the last component emits, not which
        class it happens to be. A structure ending in a threshold emits binary
        labels, which makes it a detector as far as a caller is concerned.

        Returns
        -------
        str
            One of ``"score"``, ``"labels"``, ``"series"``.
        """
        return self.terminal.output_kind

    def _parts(self) -> dict[str, Component]:
        """Name the parts, so ``set_params(part__param=...)`` reaches them."""
        raise NotImplementedError

    def _verbs(self) -> tuple[str, ...]:
        """Return the accepted verbs, most descriptive first."""
        return _VERBS[self.output_kind]

    def _require(self, verb: str) -> None:
        """Fail with a redirect when the caller used a verb that cannot apply."""
        accepted = self._verbs()
        if verb not in accepted:
            primary = accepted[0]
            msg = (
                f"This {type(self).__name__} ends in "
                f"{type(self.terminal).__name__}, which produces "
                f"{_OUTPUT_DESCRIPTION[self.output_kind]}, so call {primary}() or "
                f"fit_{primary}() rather than {verb}()."
            )
            raise TypeError(msg)

    # Every verb is offered, and the wrong one is refused with a redirect. That
    # is friendlier than exposing only `run`, because callers reach for the verb
    # matching what they built.

    def score(self, data: Any) -> Any:
        """Score ``data``; valid when this structure ends in a scorer."""
        self._require("score")
        return self._emit(data)

    def fit_score(self, data: Any) -> Any:
        """Fit on ``data`` and score it; valid when it ends in a scorer."""
        self._require("score")
        return self._fit_emit(data)

    def apply(self, data: Any) -> Any:
        """Label ``data``; valid when this structure ends in labels."""
        self._require("apply")
        return self._emit(data)

    def fit_apply(self, data: Any) -> Any:
        """Fit on ``data`` and label it; valid when it ends in labels."""
        self._require("apply")
        return self._fit_emit(data)

    def transform(self, data: Any) -> Any:
        """Transform ``data``; valid when this structure ends in a transformer."""
        self._require("transform")
        return self._emit(data)

    def fit_transform(self, data: Any) -> Any:
        """Fit on ``data`` and transform it; valid when it ends in a transformer."""
        self._require("transform")
        return self._fit_emit(data)

    def detect(self, data: Any) -> Any:
        """Detect anomalies in ``data``; valid when it ends in labels."""
        self._require("detect")
        return self._emit(data)

    def fit_detect(self, data: Any) -> Any:
        """Fit on ``data`` and detect anomalies in it.

        The usual entry point for unsupervised use, where one series both defines
        normal behaviour and is searched for departures from it.

        Parameters
        ----------
        data
            Any supported dataframe, series, or :class:`~hazure.TimeSeries`.

        Returns
        -------
        Any
            Labels, in the same flavour as the input.
        """
        self._require("detect")
        return self._fit_emit(data)


def _check_unique_names(names: Sequence[str], kind: str) -> None:
    """Refuse a structure that names two of its parts the same thing.

    Parameters
    ----------
    names
        The names, in declaration order.
    kind
        What the parts are called, for the message: ``"Step"`` or ``"Node"``.

    Raises
    ------
    ValueError
        A name appears more than once.
    """
    if len(set(names)) != len(names):
        repeated = sorted({n for n in names if names.count(n) > 1})
        msg = f"{kind} names must be unique; repeated: {repeated}."
        raise ValueError(msg)


def _check_is_component(model: Any, kind: str, name: str) -> None:
    """Refuse a part that is not something this package knows how to run.

    Parameters
    ----------
    model
        Whatever was passed in the part's place.
    kind
        What the part is called, for the message: ``"Step"`` or ``"Node"``.
    name
        The part's name.

    Raises
    ------
    TypeError
        ``model`` is not a component.
    """
    if not isinstance(model, Component):
        msg = (
            f"{kind} {name!r} holds {type(model).__name__}, which is not a "
            f"hazure component."
        )
        raise TypeError(msg)
