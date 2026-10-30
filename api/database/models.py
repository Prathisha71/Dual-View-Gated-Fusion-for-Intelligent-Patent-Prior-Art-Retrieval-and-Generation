from sqlalchemy import Column, String, Integer, Float, Text, DateTime, func
from api.database.connection import Base

class PatentRecordDB(Base):
    __tablename__ = "patents"

    patent_id = Column(String(50), primary_key=True, index=True)
    title = Column(Text, nullable=False)
    abstract = Column(Text, nullable=False)
    claims = Column(Text)
    independent_claims = Column(Text)
    primary_cpc = Column(String(20), index=True)
    cpc_subclasses = Column(String(255))
    grant_date = Column(String(20), index=True)
    cited_patent_ids = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class SearchHistoryDB(Base):
    __tablename__ = "search_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query_patent_id = Column(String(50))
    query_title = Column(Text)
    top_cpc = Column(String(20))
    top_candidate_id = Column(String(50))
    top_similarity = Column(Float)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
