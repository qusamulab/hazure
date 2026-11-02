# Compatibility fixtures

`pipeline-0.2.json` was produced before the package reorganisation from commit
`f841a1c`. It contains a fitted `StandardScale → DeviationScorer → FixedThreshold`
pipeline, its input arrays and its expected labels.

Keep this payload unchanged: loading a fixture from the previous layout checks
compatibility that a round trip through the current implementation cannot prove.
