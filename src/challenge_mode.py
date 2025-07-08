# ===============================================================
# File: src/challenge_mode.py
# Description: Handles generation and evaluation of challenge questions.
# ===============================================================

import logging
from typing import List, Any, Dict, Union 
import streamlit as st
import google.generativeai as genai 
import json 
import re   
from langchain_community.chat_models import ChatOllama 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define a type alias for the LLM, as it can be either Gemini or Ollama
LLM_TYPE = Union[genai.GenerativeModel, ChatOllama]

# Import prompt templates
from prompt_templates import CHALLENGE_QUESTION_GENERATION_PROMPT, ANSWER_EVALUATION_PROMPT

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

def generate_challenge_questions(document_content: str, llm: LLM_TYPE, num_questions: int = 3) -> List[str]: # Type hint changed
    """
    Generates logic-based challenge questions from the document content using the selected LLM.
    The document_content is now expected to come from the database.
    """
    if not document_content.strip():
        st.error("No document content available from the database to generate questions.")
        return []

    logging.info(f"Generating {num_questions} challenge questions from document content.")
    
    try:
        prompt = CHALLENGE_QUESTION_GENERATION_PROMPT.format(
            document_content=document_content,
            num_questions=num_questions
        )
        
        response_text = ""
        if isinstance(llm, genai.GenerativeModel):
            response = llm.generate_content(prompt)
            response_text = response.text
        elif isinstance(llm, ChatOllama):
            response = llm.invoke(prompt)
            response_text = response.content
        else:
            raise ValueError("Unsupported LLM type for question generation.")

        # Extract JSON from markdown if present
        json_string = _extract_json_from_markdown(response_text)
        
        # Parse the JSON string. It's expected to be a list of strings directly.
        questions_data = json.loads(json_string)
        
        # Validate that questions_data is a list and its elements are strings
        if not isinstance(questions_data, list) or not all(isinstance(q, str) for q in questions_data):
            logging.warning(f"LLM response for questions was not a list of strings as expected. Raw response: {response_text[:200]}...")
            st.warning("AI model returned questions in an unexpected format. Please try again.")
            return [f"Failed to parse questions. Raw AI response: {response_text}"]

        # Directly use the strings as questions
        questions = questions_data[:num_questions] # Take up to num_questions
        logging.info(f"Generated {len(questions)} questions.")
        return questions
    except json.JSONDecodeError:
        logging.warning(f"LLM response for questions was not valid JSON or could not be parsed. Raw response: {response_text[:200]}...")
        st.warning("AI model returned an unparseable response for questions. Please try again.")
        return [f"Failed to generate questions. Raw AI response: {response_text}"]
    except Exception as e:
        logging.error(f"❌ Error generating challenge questions with LLM: {e}", exc_info=True)
        st.error(f"Failed to generate questions. This might be due to an issue with the AI model. Error: {e}")
        return []

def evaluate_user_answer(question: str, user_answer: str, document_content: str, llm: LLM_TYPE) -> Dict[str, Any]:
    """
    Evaluates a user's answer against the document content using the selected LLM.
    The document_content is now expected to come from the database.
    """
    if not document_content.strip():
        return {
            "accuracy_score": 0,
            "completeness_score": 0,
            "clarity_score": 0,
            "feedback": "Evaluation failed: No document content available from the database."
        }
    
    logging.info(f"Evaluating user answer for question: {question[:50]}...")

    try:
        prompt = ANSWER_EVALUATION_PROMPT.format(
            document_content=document_content,
            question=question,
            user_answer=user_answer
        )

        response_text = ""
        if isinstance(llm, genai.GenerativeModel):
            response = llm.generate_content(prompt)
            response_text = response.text
        elif isinstance(llm, ChatOllama):
            response = llm.invoke(prompt)
            response_text = response.content
        else:
            raise ValueError("Unsupported LLM type for answer evaluation.")

        # Extract JSON from markdown if present
        json_string = _extract_json_from_markdown(response_text)
        
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
            logging.warning(f"LLM response for evaluation was not valid JSON or could not be parsed. Raw response: {response_text[:200]}...")
            st.warning("AI model returned an unparseable response for evaluation. Please try again.")
            # Fallback to plain text if JSON parsing fails
            return {
                "accuracy_score": 0,
                "completeness_score": 0,
                "clarity_score": 0,
                "feedback": f"Failed to parse structured evaluation. Raw AI response: {response_text}"
            }
    except Exception as e:
        logging.error(f"❌ Error evaluating user answer with LLM: {e}", exc_info=True)
        st.error(f"Failed to evaluate your answer. This might be due to an issue with the AI model. Error: {e}")
        return {
            "accuracy_score": 0,
            "completeness_score": 0,
            "clarity_score": 0,
            "feedback": f"Failed to evaluate answer: {e}"
        }
