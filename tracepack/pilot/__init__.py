"""Experiment drivers.

The SDK imports exactly two things from here: `cw_arms` (which selection and assembly each published
configuration used) and `tp_serve` (universe construction).  Everything else is the experiment
harness -- stage runners, audits, analyses -- and is not part of the public API.

It lives inside the installed package rather than outside it because `tracepack.pack()` depends on
those two modules, and an SDK whose default configuration is defined in a file the user did not
install would be reproducible only by accident.
"""
