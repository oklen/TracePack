"""The dependency-free core: graph, router, typed closure, budget assembler, excerpter.

Nothing in here imports a third-party package.  That is deliberate: the packet builder has to be
droppable into an agent loop without pulling in a model stack, and the offline dense router uses
keyed BLAKE2b rather than an embedding model for the same reason.
"""
