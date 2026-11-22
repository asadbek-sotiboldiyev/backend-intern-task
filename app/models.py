from typing import Annotated

from pydantic import BaseModel, Field, field_validator

class DocumentItem(BaseModel):
    id: Annotated[str, Field(min_length=1)]
    text: Annotated[str, Field(min_length=1)]

    @field_validator("id", "text")
    @classmethod
    def not_whitespace_only(cls, value: str, info) -> str:
        if not value.strip():
            raise ValueError(f"'{info.field_name}' must not be empty or whitespace only")
        return value


class IngestRequest(BaseModel):
    documents: Annotated[list[DocumentItem], Field(min_length=1)]


class AskRequest(BaseModel):
    question: Annotated[str, Field(min_length=1)]
    top_k: Annotated[int, Field(default=3, ge=1, le=10)]

    @field_validator("question")
    @classmethod
    def question_not_whitespace_only(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("'question' must not be empty or whitespace only")
        return value