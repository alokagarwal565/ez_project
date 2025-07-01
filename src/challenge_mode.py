# ===============================================================
# File: src/challenge_mode.py
# Description: Handles generation and evaluation of challenge questions.
# ===============================================================

import logging
from typing import List, Any
import streamlit as st
import google.generativeai as genai
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# Removed Langchain LLM imports as we're using direct genai calls
# from langchain_google_genai import ChatGoogleGenerativeAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_challenge_questions(document_content: str, llm: genai.GenerativeModel, num_questions: int = 3) -> List[str]:
    """
    Generates logic-based challenge questions from the document content using Gemini.
    """
    if not document_content or not llm:
        return []

    logging.info(f"Generating {num_questions} challenge questions.")
    question_prompt = f"""
    Based on the following document content, generate {num_questions} challenging, logic-based questions.
    These questions should require understanding and inference, not just direct recall.
    Ensure the questions are answerable from the provided text.
    Format each question with a number, e.g., "1. Question one?".

    Document Content:
    ---
    {document_content}
    ---

    Challenge Questions:
    """
    try:
        # Use direct generate_content call
        response = llm.generate_content(question_prompt)
        # Parse the response into a list of questions
        questions_raw = response.text.strip().split('\n')
        questions = [q.strip() for q in questions_raw if q.strip() and q[0].isdigit()]
        return questions[:num_questions] # Return only the requested number of questions
    except Exception as e:
        logging.error(f"❌ Error generating challenge questions with Gemini: {e}", exc_info=True)
        return [f"Failed to generate questions: {e}"]

def evaluate_user_answer(question: str, user_answer: str, document_content: str, llm: genai.GenerativeModel) -> str:
    """
    Evaluates a user's answer against the document content using Gemini.
    Provides detailed feedback.
    """
    if not question or not user_answer or not document_content or not llm:
        return "Cannot evaluate: missing question, answer, document content, or LLM."

    logging.info(f"Evaluating user answer for question: {question[:50]}...")
    evaluation_prompt = f"""
    You are an evaluator. Your task is to compare a user's answer to a question based on a given document.
    Provide constructive feedback.

    Question: "{question}"
    User's Answer: "{user_answer}"

    Document Content (for reference):
    ---
    {document_content}
    ---

    Please provide your evaluation in the following format:
    1. **Accuracy**: Is the user's answer factually correct based on the document? (Yes/No/Partially)
    2. **Completeness**: Does the user's answer address all aspects of the question that can be derived from the document? (Yes/No/Partially)
    3. **Clarity**: Is the user's answer clear and easy to understand? (Yes/No)
    4. **Justification/Feedback**: Explain why the answer is accurate/inaccurate or complete/incomplete, referencing the document where appropriate. Suggest improvements if necessary.
    5. **Score**: Assign a score from 0 to 10 for the user's answer, where 10 is excellent and 0 is completely incorrect.
    """
    try:
        # Use direct generate_content call
        response = llm.generate_content(evaluation_prompt)
        return response.text
    except Exception as e:
        logging.error(f"❌ Error evaluating user answer with Gemini: {e}", exc_info=True)
        return f"Failed to evaluate answer: {e}"
