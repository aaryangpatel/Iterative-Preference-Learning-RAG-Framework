"""CRUCIBLE-style report generation, citation tracking, and nugget alignment."""

from crucible.models.alignment import NuggetAlignmentResult
from crucible.models.citation import CitedReport, CitedSentence, Citation, SourceSpan
from crucible.models.nugget import CrucibleNugget, CrucibleNuggetBank
from crucible.nuggets.report_extractor import ReportNuggetExtractor
from crucible.alignment.report_aligner import CrucibleReportNuggetAligner
from crucible.alignment.text_aligner import CrucibleTextAligner

__all__ = [
    "CitedReport",
    "CitedSentence",
    "Citation",
    "CrucibleNugget",
    "CrucibleNuggetBank",
    "CrucibleReportNuggetAligner",
    "CrucibleTextAligner",
    "NuggetAlignmentResult",
    "ReportNuggetExtractor",
    "SourceSpan",
]

__version__ = "0.1.0"
