"""
MIMOSA
==================

This package provides a modular and extensible framework for de‑novo motif
discovery, motif comparison and performance evaluation in ChIP‑seq like
applications.  The design emphasises object oriented abstractions and clear
interfaces so that new motif discovery tools, motif comparison strategies
and evaluation metrics can be added with minimal effort.  All code is
intended for educational and fundamental research use on open, anonymised
datasets only and must not be applied to sensitive or identifiable data.

The top level modules expose the following key components:

``io``
    Functions for reading and writing biological sequences in FASTA format
    and motifs in MEME format.

``models``
    Motif model classes.  A generic :class:`MotifModel` describes the common
    behaviour of sequence motifs, while concrete subclasses such as
    :class:`PWMMotif` implement specific scoring logic.

``discovery``
    Base classes and concrete implementations of motif discovery tools.

``comparison``
    Base classes and concrete implementations of motif comparison metrics.

``evaluation``
    Classes for computing ROC/PR curves and their associated summary
    statistics.

``pipeline``
    High level orchestrators that perform bootstrapping, odd/even
    cross‑validation, motif comparison and final motif selection.

``cli``
    An example command line interface exposing the pipeline to end users.

This modular structure makes it straightforward to integrate alternative
motif models (e.g. BaMM or SiteGA), comparison methods (e.g. Jaccard
index, overlap coefficient or recognition‑function correlation) and
performance metrics.  See individual modules for detailed API
documentation.
"""
