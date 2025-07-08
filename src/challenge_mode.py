# ===============================================================
# File: src/challenge_mode.py
# Description: Handles generation and evaluation of challenge questions.
# ===============================================================

import logging
from typing import List, Any, Dict
import streamlit as st
import google.generativeai as genai
import json # Import json for parsing LLM output
import re   # Import re for regex to strip markdown

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _extract_json_from_markdown(text: str) -> str:
    """
    Extracts a JSON string from a markdown code block.
    Assumes the JSON is within ```json ... ``` or ``` ... ```
    """
    # Regex to find content within ```json ... ``` or ``` ... ```
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip() # Return original text if no markdown block found

def generate_challenge_questions(document_content: str, llm: genai.GenerativeModel, num_questions: int = 3) -> List[str]:
    """
    Generates logic-based challenge questions from the document content using Gemini.
    """
    if not document_content or not llm:
        st.warning("Cannot generate questions: missing document content or AI model not initialized.")
        return []

    logging.info(f"Generating {num_questions} challenge questions.")

    # Use the structured prompt from prompt_templates.py
    from prompt_templates import CHALLENGE_QUESTION_GENERATION_PROMPT
    question_prompt = CHALLENGE_QUESTION_GENERATION_PROMPT.format(
        document_content=document_content
    )

    try:
        response = llm.generate_content(question_prompt)
        raw_response_text = response.text

        # First, try to extract JSON from markdown if present
        json_string = _extract_json_from_markdown(raw_response_text)

        # Attempt to parse the JSON response
        try:
            questions_json = json.loads(json_string)
            if isinstance(questions_json, list) and all(isinstance(q, str) for q in questions_json):
                questions = questions_json
            else:
                logging.warning(f"LLM returned JSON but not a list of strings: {json_string[:100]}...")
                st.warning("AI model returned questions in an unexpected format. Displaying raw questions.")
                raise ValueError("LLM did not return a valid JSON list of strings.")
        except json.JSONDecodeError:
            logging.warning(f"LLM response for questions was not valid JSON or could not be parsed as such. Raw response: {raw_response_text[:200]}...")
            st.warning("AI model returned an unparseable response for questions. Displaying raw questions.")
            # Fallback to line-by-line parsing if JSON parsing fails
            questions_raw = raw_response_text.strip().split('\n')
            questions = [q.strip() for q in questions_raw if q.strip() and (q[0].isdigit() or q.startswith('- '))] # Also handle markdown list

        return questions[:num_questions] # Return only the requested number of questions
    except Exception as e:
        logging.error(f"❌ Error generating challenge questions with Gemini: {e}", exc_info=True)
        st.error(f"Failed to generate challenge questions. This might be due to an issue with the AI model. Error: {e}")
        return [f"Failed to generate questions: {e}"]

def evaluate_user_answer(question: str, user_answer: str, document_content: str, llm: genai.GenerativeModel) -> Dict[str, Any]:
    """
    Evaluates a user's answer against the document content using Gemini.
    Provides detailed feedback and rubric scores.
    Returns a dictionary with scores and feedback.
    """
    if not question or not user_answer or not document_content or not llm:
        st.warning("Cannot evaluate answer: missing input (question, user answer, document content, or AI model).")
        return {"accuracy_score": 0, "completeness_score": 0, "clarity_score": 0, "feedback": "Cannot evaluate: missing input."}

    logging.info(f"Evaluating user answer for question: {question[:50]}...")

    # Use the structured prompt from prompt_templates.py
    from prompt_templates import ANSWER_EVALUATION_PROMPT
    structured_evaluation_prompt = ANSWER_EVALUATION_PROMPT.format(
        document_content=document_content,
        question=question,
        user_answer=user_answer
    )

    try:
        response = llm.generate_content(structured_evaluation_prompt)
        raw_response_text = response.text

        # Extract JSON string from markdown code block if present
        json_string = _extract_json_from_markdown(raw_response_text)

        # Attempt to parse the JSON response
        try:
            evaluation_data = json.loads(json_string)
            # Ensure all expected keys are present, provide defaults if not
            return {
                "accuracy_score": evaluation_data.get("accuracy_score", 0),
                "completeness_score": evaluation_data.get("completeness_score", 0),
                "clarity_score": evaluation_data.get("clarity_score", 0),
                "feedback": evaluation_data.get("feedback", "No detailed feedback provided.")
            }
        except json.JSONDecodeError:
            logging.warning(f"LLM response for evaluation was not valid JSON or could not be parsed. Raw response: {raw_response_text[:200]}...")
            st.warning("AI model returned an unparseable response for evaluation. Please try again.")
            # Fallback to plain text if JSON parsing fails
            return {
                "accuracy_score": 0,
                "completeness_score": 0,
                "clarity_score": 0,
                "feedback": f"Failed to parse structured evaluation. Raw AI response: {raw_response_text}"
            }
    except Exception as e:
        logging.error(f"❌ Error evaluating user answer with Gemini: {e}", exc_info=True)
        st.error(f"Failed to evaluate your answer. This might be due to an issue with the AI model. Error: {e}")
        return {
            "accuracy_score": 0,
            "completeness_score": 0,
            "clarity_score": 0,
            "feedback": f"Failed to evaluate answer: {e}"
        }
