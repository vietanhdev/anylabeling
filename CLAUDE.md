# CLAUDE.md

@AGENT.md

Claude Code must follow the shared repository guidance in `AGENT.md`. In
particular, work in a dedicated environment, add a regression test for every
bug fix, run the full headless test suite before handing off changes, and do
not tag a release until all cross-platform release checks are green.

When a change touches Qt behavior, frozen binaries, model loading, or an ONNX
Runtime provider, also complete the relevant manual checks described in
`AGENT.md`; unit tests alone are not sufficient for those paths.
