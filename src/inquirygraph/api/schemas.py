from typing import Literal

from pydantic import BaseModel, Field


class QueryUnderstanding(BaseModel):
    summary: str = Field(description="One-sentence restatement of the user's question")
    key_entities: list[str] = Field(description="Main entities to research (products, vendors, concepts)")
    dimensions: list[str] = Field(description="Topics to cover, e.g. cost, scalability, migration")
    success_criteria: str = Field(description="What a good answer must include")


class ResearchTask(BaseModel):
    id: str
    question: str
    search_query: str
    priority: Literal["high", "medium", "low"] = "medium"


class ResearchPlan(BaseModel):
    tasks: list[ResearchTask]


class ExtractedClaim(BaseModel):
    text: str
    theme: str
    entities: list[str] = Field(default_factory=list)
    evidence_excerpt: str = ""


class ExtractionResult(BaseModel):
    entities: list[str] = Field(default_factory=list, max_length=20)
    claims: list[ExtractedClaim] = Field(default_factory=list, max_length=12)


class Citation(BaseModel):
    source_title: str
    source_url: str
    excerpt: str


class ReportSection(BaseModel):
    title: str
    content: str
    citations: list[Citation] = Field(default_factory=list)


class InvestigationReport(BaseModel):
    executive_summary: str
    sections: list[ReportSection]
    contradictions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class DocumentUploadResponse(BaseModel):
    investigation_id: str
    filename: str
    chunks_indexed: int


class CoverageDecision(BaseModel):
    needs_more_research: bool
    reason: str
    follow_up_queries: list[str] = Field(default_factory=list)


class Contradiction(BaseModel):
    claim_a: str
    claim_b: str
    explanation: str
    source_a: str = ""
    source_b: str = ""


class ContradictionList(BaseModel):
    contradictions: list[Contradiction] = Field(default_factory=list)


class CitationVerification(BaseModel):
    citation_index: int
    supported: bool
    reason: str


class CitationVerificationResult(BaseModel):
    results: list[CitationVerification] = Field(default_factory=list)
