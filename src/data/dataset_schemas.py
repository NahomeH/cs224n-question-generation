"""
Schema definitions for datasets used in the medical question generation project.

This module defines pydantic models that represent the structure of
datasets used for training and evaluation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union


class ClinicalCase(BaseModel):
    """Schema for a clinical case description."""
    
    text: str = Field(..., description="The clinical case description")
    meta: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Optional metadata about the case"
    )


class USMLEQuestion(BaseModel):
    """Schema for a USMLE-style question."""
    
    question: str = Field(..., description="The question text")
    answer: Optional[str] = Field(
        default=None, 
        description="Answer to the question (if available)"
    )
    options: Optional[List[str]] = Field(
        default=None, 
        description="Multiple choice options (if available)"
    )
    explanation: Optional[str] = Field(
        default=None, 
        description="Explanation of the correct answer (if available)"
    )
    difficulty: Optional[str] = Field(
        default=None, 
        description="Difficulty level (e.g., 'easy', 'medium', 'hard')"
    )
    tags: Optional[List[str]] = Field(
        default=None, 
        description="Medical topic tags"
    )


class MedicalQAPair(BaseModel):
    """Schema for a paired clinical case and question."""
    
    clinical_case: Union[str, ClinicalCase] = Field(
        ..., 
        description="The clinical case (can be a string or ClinicalCase object)"
    )
    question: Union[str, USMLEQuestion] = Field(
        ..., 
        description="The USMLE question (can be a string or USMLEQuestion object)"
    )
    id: Optional[str] = Field(
        default=None, 
        description="Unique identifier for the pair"
    )
    source: Optional[str] = Field(
        default=None, 
        description="Source of the data"
    )


class MedicalDataset(BaseModel):
    """Schema for a complete medical QA dataset."""
    
    pairs: List[MedicalQAPair] = Field(
        ..., 
        description="List of clinical case-question pairs"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Dataset metadata"
    )
    version: Optional[str] = Field(
        default="1.0", 
        description="Version of the dataset"
    )


def validate_dataset(dataset: Dict[str, Any]) -> MedicalDataset:
    """
    Validate a dataset against the schema.
    
    Args:
        dataset: Dictionary representation of the dataset
        
    Returns:
        Validated MedicalDataset object
    
    Raises:
        ValidationError: If the dataset doesn't match the schema
    """
    return MedicalDataset(**dataset) 