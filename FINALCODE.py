import requests
import spacy
from sympy import solve, symbols, sympify, SympifyError
from huggingface_hub import InferenceClient
import easyocr
import streamlit as st
import os
import cv2
import numpy as np
import re
import json
try:
    import pytesseract
except ImportError:
    pytesseract = None

# Define API keys
OPENROUTER_API_KEY = "MYAPIKEY"  # OpenRouter API key for DeepSeek
HF_API_KEY = "MYAPIKEY"  # Hugging Face API token
SERPAPI_KEY = "MYAPIKEY"  # SerpAPI key for current events

# Initialize Hugging Face client
hf_client = InferenceClient(api_key=HF_API_KEY)

# Initialize spaCy for NLP-based question classification
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("spaCy model 'en_core_web_sm' not found. Run 'python -m spacy download en_core_web_sm'.")
    st.stop()

# Initialize EasyOCR for image text extraction (CPU-only for zero-cost)
ocr_reader = easyocr.Reader(['en'], gpu=False)

# Set Tesseract executable path (adjust as needed for your system)
if pytesseract:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Cache for API responses to stay within free-tier limits
response_cache = {}

def preprocess_image(image_path):
    """Preprocess image for better OCR accuracy."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite('preprocessed.jpg', otsu)
        return 'preprocessed.jpg'
    except Exception as e:
        return f"Error: Failed to preprocess image ({e})."

def extract_text_with_tesseract(image_path):
    """Extract text using Tesseract OCR as a fallback."""
    if not pytesseract:
        return "Error: pytesseract is not installed."
    try:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Enhance contrast for better recognition
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        text = pytesseract.image_to_string(enhanced, config='--psm 6')  # Assume single uniform block of text
        return text.strip()
    except Exception as e:
        return f"Error: Tesseract failed to extract text ({e})."

def parse_question(question):
    """Identify question type (math, grammar, general, mcq, fill-in-the-blank) using spaCy."""
    if not question:
        return "general"
    doc = nlp(question.lower())
    # Math: Check for operators or keywords
    if any(token.text in ["+", "-", "*", "/", "=", "solve", "calculate"] for token in doc):
        return "math"
    # Grammar: Check for correction keywords
    if any(word in question.lower() for word in ["correct", "grammar", "fix"]):
        return "grammar"
    # MCQ: Check for options (e.g., A), B), C))
    if re.search(r'\b[A-D]\)\s', question, re.IGNORECASE):
        return "mcq"
    # Fill-in-the-blank: Check for underscores or blanks
    if "____" in question or "_____" in question:
        return "fill-in-the-blank"
    # General: Default for other questions
    return "general"

def is_current_event_question(question):
    """Check if the question is about current events."""
    keywords = ["won", "champion", "match", "tournament", "cricket", "ipl", "latest", "current", "recent"]
    return any(keyword in question.lower() for keyword in keywords)

def web_search(query):
    """Perform a web search using SerpAPI for current event questions."""
    if not SERPAPI_KEY:
        return "Error: SerpAPI key is not configured. Please provide a valid key or contact support."
    try:
        response = requests.get(
            "https://serpapi.com/search",
            params={
                "q": query,
                "api_key": SERPAPI_KEY,
                "engine": "google",
                "hl": "en",  # Language setting
                "gl": "in"   # Country setting for India
            },
            timeout=10
        )
        response.raise_for_status()
        results = response.json().get("organic_results", [])
        if not results:
            return "No relevant results found."
        relevant_results = [
            {"content": r.get("snippet", "").lower()}
            for r in results
            if any(keyword in r.get("snippet", "").lower() for keyword in ["won", "champion", "match", "cricket", "ipl"])
        ]
        return relevant_results if relevant_results else [{"content": results[0].get("snippet", "No relevant content")}]
    except requests.RequestException as e:
        return f"Error: Web search failed ({str(e)}). Please try again later."
    except json.JSONDecodeError:
        return "Error: Unable to parse search results."
    except Exception as e:
        return f"Error: Unexpected issue during web search ({str(e)})."

def solve_math(question):
    """Solve math problems, first with SymPy, then OpenRouter DeepSeek API if needed."""
    try:
        x = symbols("x")
        expr = question.replace("=", "-").replace("solve", "").strip()
        solution = solve(sympify(expr), x)
        if solution:
            return f"Solution: x = {solution[0]}\nExplanation: Solved algebraically using symbolic computation."
    except (SympifyError, ValueError, TypeError):
        pass
    
    cache_key = f"math_{question}"
    if cache_key in response_cache:
        return response_cache[cache_key]
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json={
                "model": "deepseek/deepseek-chat",
                "messages": [{"role": "user", "content": f"Solve with step-by-step explanation: {question}"}],
                "stream": False
            },
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        answer = result["choices"][0]["message"]["content"]
        response_cache[cache_key] = answer
        return answer
    except requests.RequestException as e:
        return f"Error: Failed to connect to OpenRouter API ({e}). Please try again."
def solve_grammar(question):
    """Correct grammar using Hugging Face LLaMA, with explanation."""
    cache_key = f"grammar_{question}"
    if cache_key in response_cache:
        return response_cache[cache_key]
    
    try:
        result = hf_client.text_generation(
            prompt=f"Correct the following sentence and explain the corrections: {question}",
            model="meta-llama/Llama-3.1-8B",
            max_new_tokens=150
        )
        response_cache[cache_key] = result
        return result
    except Exception as e:
        return f"Error: Failed to process grammar correction ({e}). Please try again."

def solve_mcq(question, context=None):
    """Handle MCQ questions by parsing options and selecting the correct one."""
    cache_key = f"mcq_{question}_{context or ''}"
    if cache_key in response_cache:
        return response_cache[cache_key]
    
    try:
        options = re.findall(r'([A-D]\)\s.*?)(?=[A-D]\)|$)', question, re.IGNORECASE)
        question_text = re.split(r'[A-D]\)\s', question)[0].strip()
        if context:
            prompt = f"Question: {question_text}\nOptions: {', '.join(options)}\nContext: {context}\nSelect the correct option and explain."
        else:
            prompt = f"Question: {question_text}\nOptions: {', '.join(options)}\nSelect the correct option and explain."
        result = hf_client.text_generation(
            prompt=prompt,
            model="meta-llama/Llama-3.1-8B",
            max_new_tokens=200
        )
        response_cache[cache_key] = result
        return result
    except Exception as e:
        return f"Error: Failed to process MCQ ({e}). Please try again."

def solve_fill_in_the_blank(question, context=None):
    """Handle fill-in-the-blank questions using LLaMA."""
    cache_key = f"fill_{question}_{context or ''}"
    if cache_key in response_cache:
        return response_cache[cache_key]
    
    try:
        prompt = f"Fill in the blank with the correct word: {question}\nContext: {context or 'No context provided.'}\nProvide the completed sentence and an explanation."
        result = hf_client.text_generation(
            prompt=prompt,
            model="meta-llama/Llama-3.1-8B",
            max_new_tokens=150
        )
        response_cache[cache_key] = result
        return result
    except Exception as e:
        return f"Error: Failed to process fill-in-the-blank ({e}). Please try again."

def solve_general(question, context=None):
    """Answer general questions, using web search for current events."""
    if is_current_event_question(question):
        search_results = web_search(query=question)
        if isinstance(search_results, list) and search_results:
            content = search_results[0].get("content", "").lower()
            if "won" in content and "ipl" in content:
                parts = content.split("won")
                if len(parts) > 1:
                    team = parts[1].split()[0].title()
                    answer = f"Answer: The {team} won the IPL.\nExplanation: Fetched from web search."
                    response_cache[f"general_{question}_{context or ''}"] = answer
                    return answer
            return search_results[0].get("content", "No relevant content found.")
        return search_results if search_results else "Error: Web search unavailable or no relevant data found."
    else:
        cache_key = f"general_{question}_{context or ''}"
        if cache_key in response_cache:
            return response_cache[cache_key]
        
        try:
            result = hf_client.question_answering(
                question=question,
                context=context or question,
                model="deepset/roberta-base-squad2"
            )
            answer = result["answer"]
            response_cache[cache_key] = f"Answer: {answer}\nExplanation: Extracted from context using question-answering model."
            return response_cache[cache_key]
        except Exception as e:
            return f"Error: Failed to process question ({e}). Please try again."

def solve_question(question, context=None, image_path=None):
    """Main function to solve questions, handling image input if provided."""
    if image_path:
        try:
            preprocessed_path = preprocess_image(image_path)
            if not preprocessed_path:
                return "Error: Failed to load image."
            extracted_text = ocr_reader.readtext(preprocessed_path, detail=0)
            os.remove(preprocessed_path)
            question = extracted_text[0] if extracted_text else question
            # Validate extracted text for math
            if not any(op in question for op in ["+", "-", "*", "/", "="]):
                # Try Tesseract as fallback
                tesseract_text = extract_text_with_tesseract(image_path)
                if isinstance(tesseract_text, str) and any(op in tesseract_text for op in ["+", "-", "*", "/", "="]):
                    question = tesseract_text
                else:
                    return f"Error: Invalid math expression detected in image. Please upload a clearer image.\nExtracted text: {question}\nTesseract text: {tesseract_text}"
        except Exception as e:
            return f"Error: Failed to extract text from image ({e})."
    
    question_type = parse_question(question)
    
    if question_type == "math":
        return solve_math(question)
    elif question_type == "grammar":
        return solve_grammar(question)
    elif question_type == "mcq":
        return solve_mcq(question, context)
    elif question_type == "fill-in-the-blank":
        return solve_fill_in_the_blank(question, context)
    else:
        return solve_general(question, context)

def main():
    """Streamlit web app for Auto-Solver, with input forms and sample questions."""
    st.set_page_config(page_title="Auto-Solver", page_icon="🧠")
    st.title("Auto-Solver: AI-Powered Question Solver")
    st.write("Solve math, grammar, MCQs, fill-in-the-blank, and general questions with explanations!")
    
    # Separate placeholders for math and general questions
    st.header("Math Problem")
    math_question = st.text_input("Enter a math problem (e.g., '2x + 5 = 15')")
    math_image = st.file_uploader("Optional: Upload a math problem image", type=["jpg", "png"], key="math_image")
    
    st.header("General, MCQ, or Fill-in-the-Blank Question")
    general_question = st.text_input("Enter a general, MCQ, or fill-in-the-blank question (e.g., 'What is the capital of France?', 'What is 2+2? A) 3 B) 4 C) 5 D) 6', 'The capital of France is ____')")
    context = st.text_area("Optional: Provide context for general/MCQ/fill-in-the-blank questions", placeholder="e.g., France is in Europe. Its capital is Paris.")
    
    if st.button("Solve", key="solve_button"):
        if not math_question and not general_question and not math_image:
            st.error("Please provide a question or upload an image.")
        else:
            with st.spinner("Solving..."):
                if math_image:
                    with open("temp.jpg", "wb") as f:
                        f.write(math_image.getbuffer())
                    result = solve_question(math_question, context, image_path="temp.jpg")
                    if os.path.exists("temp.jpg"):
                        os.remove("temp.jpg")
                elif math_question:
                    result = solve_question(math_question, context)
                else:
                    result = solve_question(general_question, context)
                st.markdown("**Result:**")
                st.write(result)
    
    st.sidebar.header("Try Sample Questions")
    sample_questions = [
        "Solve: 2x + 5 = 15",
        "Correct: He go to school",
        "What is the capital of France? (Context: France is in Europe. Its capital is Paris.)",
        "Who won the IPL cricket match in 2020?",
        "What is 2+2? A) 3 B) 4 C) 5 D) 6 (Context: Basic arithmetic)",
        "The capital of France is ____ (Context: France is in Europe. Its capital is Paris.)"
    ]
    for sq in sample_questions:
        if st.sidebar.button(sq):
            st.session_state["math_question"] = sq if "Solve:" in sq else ""
            st.session_state["general_question"] = sq if "Solve:" not in sq else ""
            st.session_state["context"] = sq.split("Context:")[1][:-1] if "Context:" in sq else ""
            st.experimental_rerun()

if __name__ == "__main__":
    main()