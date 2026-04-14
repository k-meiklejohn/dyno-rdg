"""
RNASequence - A BioPython-backed class for RNA transcript analysis.

Inherits from Bio.SeqRecord.SeqRecord, so all standard BioPython methods
(IO, alignment, feature annotation, slicing, …) work out of the box.

Custom additions:
    - find_orfs()       : locate all open reading frames
    - translate_orfs()  : translate all ORFs to protein sequences
    - find_motifs()     : search for sequence motifs (string or regex)
    - gc_content()      : GC percentage
    - nucleotide_freq() : per-base frequency counts
"""

from __future__ import annotations
import re
from dataclasses import dataclass

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .events import RiboEvent
from ..simulation import TransitionMap, RiboGraphFlux
from ..viz import RiboGraphVis
import pandas as pd
import Levenshtein as lv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

AUG_SCORE = pd.read_csv(BASE_DIR / 'aug.csv')
NON_AUG_SCORE = pd.read_csv(BASE_DIR / 'non_aug.csv')


def start_score(sequence: str, aug: bool):
    df = AUG_SCORE if aug else NON_AUG_SCORE
    
    match = df[df['sequence'] == sequence]
    
    if match.empty:
        return 0  # or raise an error / default value
    
    efficiency = match['efficiency'].values[0]
    return min(efficiency * 0.84/100, 1)
    
# ---------------------------------------------------------------------------
# Helper dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MotifMatch:
    """A single motif hit within an RNA sequence."""
    motif: str    # the original query string / pattern
    start: int
    end: int
    matched: str  # actual matched substring (differs from motif for regex)

    def __repr__(self) -> str:
        return (
            f"MotifMatch(motif='{self.motif}', "
            f"start={self.start}, end={self.end}, matched='{self.matched}')"
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RNASequence(SeqRecord):
    """
    An RNA transcript record with custom analysis methods.

    Inherits all BioPython SeqRecord functionality (reading/writing FASTA,
    GenBank, feature annotation, slicing, …) and adds RNA-specific helpers.

    Parameters
    ----------
    sequence : str | Bio.Seq.Seq
        The RNA sequence. DNA (T) is silently converted to RNA (U).
    id : str
        Transcript identifier (e.g. "ENST00000123456").
    name : str
        Short name (e.g. gene symbol).
    description : str
        Free-text description.
    **kwargs
        Any additional keyword arguments accepted by SeqRecord.

    Examples
    --------
    >>> rna = RNASequence("AUGCUACGAUAA", id="tx001", name="demo")
    >>> rna.gc_content()
    0.5
    >>> orfs = rna.find_orfs(min_length=3)
    >>> orfs[0].protein
    'MLR'
    """
    def __init__(
        self,
        sequence: str | Seq,
        id: str = "<unknown id>",
        name: str = "<unknown name>",
        description: str = "<unknown description>",
        *args,
        **kwargs,
    ) -> None:

        raw = str(sequence).upper().replace("T", "U")
        bad = set(raw) - set("AUCGN")
        if bad:
            raise ValueError(f"Invalid RNA characters detected: {bad}")

        # Only set defaults if not already provided
        kwargs.setdefault("id", id)
        kwargs.setdefault("name", name)
        kwargs.setdefault("description", description)

        super().__init__(Seq(raw), *args, **kwargs)

        self.events: list[RiboEvent] = []
        self.auto_find_events()
    
    @property
    def transition_map(self) -> TransitionMap:
        list_of_transitions = []
        for event in self.events:
            list_of_transitions.extend(event._to_transition())
        tmap = TransitionMap()
        tmap.add_weighted_edges_from(list_of_transitions)
        return tmap


    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.seq)

    def __str__(self) -> str:
        preview = str(self.seq)[:40]
        ellipsis = "..." if len(self) > 40 else ""
        return (
            f"RNASequence(id={self.id!r}, name={self.name!r}, "
            f"len={len(self)}, seq={preview}{ellipsis})"
        )

    def __getitem__(self, index):
        parent = super().__getitem__(index)
        
        new = RNASequence(
            str(parent.seq),
            id=parent.id,
            name=parent.name,
            description=parent.description,
        )
        
        return new


    def auto_find_events(self):
        for i in range(6, len(self)):
            codon = str(self.seq[i:i+3])
            if codon == 'AUG':
                self.events.append(RiboEvent(i+1, 'initiation', start_score(sequence=str(self.seq[i-6:i+5]), aug=True), 0 ))
            elif lv.distance(codon, 'AUG') == 1:
                if start_score(sequence=str(self.seq[i-4:i+4]), aug=False) >= 0.001:
                    self.events.append(RiboEvent(i+1, 'initiation', start_score(sequence=str(self.seq[i-4:i+4]), aug=False), 0))

            elif codon in ['UAG', 'UGA', 'UAA']:
                self.events.append(RiboEvent(i+1, 'termination', 0, 1))
        self.events.append(RiboEvent(1, 'cap', 1, 0))
        self.events.append(RiboEvent(len(self.seq), 'end', 0, 1))


            

        





