import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import csv
import pandas as pd
from datetime import datetime
from datetime import time as dt_time_constructor
import time as timestamp_module
from streamlit_calendar import calendar
from ics import Calendar, Event
import logging
from typing import Dict, List, Optional, Union, Tuple
import json
from pathlib import Path
import PyPDF2
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import io
import base64
import tempfile
import pycountry
import gettext
from babel import Locale
import re # Add re import for parsing AI responses
import uuid
from mindshield_core.retrieval import WindowRetriever
from mindshield_core.chat_engine import ChatEngine

# Initialize retriever/chat engine once
RETRIEVER = WindowRetriever(index=[])
CHAT_ENGINE = ChatEngine(RETRIEVER)

# --- Utility Functions ---
def get_locale_dir():
    # This function now correctly points to 'locales' inside 'mindshield_streamlit'
    return os.path.join(os.path.dirname(__file__), 'locales')

# --- Language and Translation Setup ---
# Moved LANGUAGES definition here to be globally accessible
# SUPPORTED_LANGUAGES = ['en', 'es', 'fr', 'de', 'ja', 'it', 'pt', 'zh_CN'] # This will be derived from CONFIG
# LANGUAGE_NAMES = { # This will be derived from CONFIG
#     'en': "English",
#     'es': "Español (Spanish)",
#     'fr': "Français (French)",
#     'de': "Deutsch (German)",
#     'ja': "日本語 (Japanese)",
#     'it': "Italiano (Italian)",
#     'pt': "Português (Portuguese)",
#     'zh_CN': "简体中文 (Simplified Chinese)"
# }

# Configuration (ensure this is defined before LANGUAGES initialization)
CONFIG = {
    "DEFAULT_LANGUAGE": "en",
    # Ensure these language codes have corresponding .mo files in locales/<lang_code>/LC_MESSAGES/messages.mo
    "SUPPORTED_LANGUAGES": ["en", "fr", "es", "de", "ja", "hi", "ur"], # Added hi (Hindi) and ur (Urdu)
    "LANGUAGE_NAMES": {
        "en": "English",
        "fr": "Français (French)",
        "es": "Español (Spanish)",
        "de": "Deutsch (German)",
        "ja": "日本語 (Japanese)",
        "hi": "हिन्दी (Hindi)", # Added Hindi
        "ur": "اردو (Urdu)"  # Added Urdu
        # Add names for other supported languages if needed, e.g., ar, it, pt, zh_CN
        # "ar": "العربية",
        # "it": "Italiano (Italian)",
        # "pt": "Português (Portuguese)",
        # "zh_CN": "简体中文 (Simplified Chinese)"
    }
}

# Initialize translations for all supported languages from CONFIG
LANGUAGES = {}
# Use language codes from CONFIG for loading translations
for lang_code in CONFIG["SUPPORTED_LANGUAGES"]:
    try:
        translation = gettext.translation(
            'messages',  # Domain
            localedir=get_locale_dir(),
            languages=[lang_code],
            fallback=True # Important: fallback to parent language (e.g. fr from fr_CA) or NullTranslations
        )
        LANGUAGES[lang_code] = translation
    except FileNotFoundError:
        # Fallback to a NullTranslations object if a .mo file is missing
        LANGUAGES[lang_code] = gettext.NullTranslations()
        if lang_code != CONFIG["DEFAULT_LANGUAGE"]: # Don't log for default lang if it's missing, as it might be source
            logger.warning(f"Translation file for '{lang_code}' not found in {get_locale_dir()}. Using fallback (English strings).")

# Ensure default language has a valid translation object (even if NullTranslations)
if CONFIG["DEFAULT_LANGUAGE"] not in LANGUAGES or not LANGUAGES[CONFIG["DEFAULT_LANGUAGE"]]:
    logger.warning(f"Default language '{CONFIG['DEFAULT_LANGUAGE']}' translation not found or invalid. Ensure .mo file exists or it will default to untranslated strings.")
    LANGUAGES[CONFIG["DEFAULT_LANGUAGE"]] = gettext.NullTranslations()

# Load environment variables
load_dotenv()

# Initialize paths
DATA_DIR = Path(__file__).parent
FEEDBACK_DIR = DATA_DIR / "feedback"
LOGS_DIR = DATA_DIR / "logs"
DATA_LIBRARY_DIR = DATA_DIR / "data_library"

# Create directories if they don't exist
for dir_path in [FEEDBACK_DIR, LOGS_DIR, DATA_LIBRARY_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Check required environment variables
required_env_vars = {
    'OPENAI_API_KEY': 'OpenAI API key for AI functionality'
}

missing_vars = []
for var, description in required_env_vars.items():
    if not os.getenv(var):
        missing_vars.append(f"{var} ({description})")

if missing_vars:
    logger.error("Missing required environment variables:\n" + "\n".join(missing_vars))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="MindShield - Mental Health Support",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load local CSS file
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Updated Translation function
def _(text: str) -> str:
    """Translates text using the gettext object in session_state."""
    if hasattr(st, 'session_state') and 'gettext_translations' in st.session_state and st.session_state.gettext_translations:
        return st.session_state.gettext_translations.gettext(text)
    
    # Fallback if session_state or translations aren't fully initialized
    # This attempts to use the default language or ultimately returns the original text.
    # CONFIG might not be defined yet if this is called extremely early.
    default_lang_code = "en" # Safe default
    if 'CONFIG' in globals() and "DEFAULT_LANGUAGE" in CONFIG:
        default_lang_code = CONFIG["DEFAULT_LANGUAGE"]
    
    if default_lang_code in LANGUAGES and LANGUAGES[default_lang_code]:
        return LANGUAGES[default_lang_code].gettext(text)
    return text # Ultimate fallback

# Define ALL exercise rendering functions first
def render_box_breathing():
    """Render an interactive box breathing exercise."""
    # Animation CSS is now in style.css
    
    st.markdown('<div class="breath-circle"></div>', unsafe_allow_html=True)
    
    phase = int(timestamp_module.time() % 16 // 4)
    phases = ["Inhale", "Hold", "Exhale", "Hold"]
    st.markdown(f"### {phases[phase]}")
    
    progress = int(timestamp_module.time() % 4 * 25)
    st.progress(progress)
    
    st.markdown("""
        Follow the circle:
        - Expands = Inhale (4s)
        - Stays large = Hold (4s)
        - Shrinks = Exhale (4s)
        - Stays small = Hold (4s)
    """)

def render_grounding():
    """Render the 5-4-3-2-1 grounding exercise."""
    if "grounding_step" not in st.session_state:
        st.session_state.grounding_step = 0
        st.session_state.grounding_items = {
            "see": [], "feel": [], "hear": [], "smell": [], "taste": []
        }
    
    steps = [
        ("see", 5, "👀 What do you see?"),
        ("feel", 4, "✋ What do you feel?"),
        ("hear", 3, "👂 What do you hear?"),
        ("smell", 2, "👃 What do you smell?"),
        ("taste", 1, "👅 What do you taste?")
    ]
    
    current = steps[st.session_state.grounding_step]
    sense, count, prompt = current
    
    st.markdown(f"### {prompt}")
    st.markdown(f"Name {count} things you can {sense}:")
    
    # Show current items
    for i, item in enumerate(st.session_state.grounding_items[sense]):
        st.text(f"{i+1}. {item}")
    
    # Input for new item
    if len(st.session_state.grounding_items[sense]) < count:
        item = st.text_input(
            "Enter item",
            key=f"grounding_{sense}_{len(st.session_state.grounding_items[sense])}",
            label_visibility="collapsed"
        )
        if item:
            st.session_state.grounding_items[sense].append(item)
            st.rerun()
    
    # Move to next step if current is complete
    if len(st.session_state.grounding_items[sense]) == count:
        if st.session_state.grounding_step < len(steps) - 1:
            if st.button(_("Next sense")):
                st.session_state.grounding_step += 1
                st.rerun()
        else:
            st.success(_("Exercise complete! How do you feel now?"))
            if st.button(_("Start 5-4-3-2-1 Over")):
                st.session_state.grounding_step = 0
                st.session_state.grounding_items = {
                    "see": [], "feel": [], "hear": [], "smell": [], "taste": []
                }
                st.rerun()
    
    # Add a button to clear all current grounding items at any stage
    if any(st.session_state.grounding_items.values()): # Show only if there are items
        if st.button(_("Clear My Grounding Items"), key="clear_grounding_items"):
            st.session_state.grounding_items = {
                "see": [], "feel": [], "hear": [], "smell": [], "taste": []
            }
            # Optionally reset step as well, or let user continue current step with cleared items
            # st.session_state.grounding_step = 0 # Uncomment to fully reset
            st.rerun()

def render_pmr():
    """Render the Progressive Muscle Relaxation exercise."""
    muscle_groups = [
        "Feet", "Calves", "Thighs", "Glutes",
        "Abdomen", "Chest", "Arms", "Hands",
        "Shoulders", "Neck", "Face"
    ]
    
    if "pmr_step" not in st.session_state:
        st.session_state.pmr_step = 0
        st.session_state.pmr_phase = "tense"  # or "relax"
        st.session_state.pmr_timer = timestamp_module.time()
    
    current_muscle = muscle_groups[st.session_state.pmr_step]
    
    # Progress bar for overall exercise
    progress = (st.session_state.pmr_step / len(muscle_groups)) * 100
    st.progress(progress)
    
    st.markdown(f"### Focus on your {current_muscle}")
    
    if st.session_state.pmr_phase == "tense":
        st.markdown("🔴 **Tense** these muscles now")
        seconds_left = 5 - int(timestamp_module.time() - st.session_state.pmr_timer)
        if seconds_left > 0:
            st.markdown(f"Hold for {seconds_left} seconds...")
        else:
            st.session_state.pmr_phase = "relax"
            st.session_state.pmr_timer = timestamp_module.time()
            st.rerun()
    else:  # relax phase
        st.markdown("💚 **Relax** and feel the tension flow away")
        seconds_left = 10 - int(timestamp_module.time() - st.session_state.pmr_timer)
        if seconds_left > 0:
            st.markdown(f"Enjoy the relaxation for {seconds_left} seconds...")
        else:
            if st.session_state.pmr_step < len(muscle_groups) - 1:
                st.session_state.pmr_step += 1
                st.session_state.pmr_phase = "tense"
                st.session_state.pmr_timer = timestamp_module.time()
            else:
                st.success("Exercise complete! Take a moment to enjoy the relaxation.")
            st.rerun()

def render_stress_checkin():
    """Render the Likert Stress Check-In exercise."""
    st.markdown("""
        Track your stress levels and identify patterns to better manage your well-being.
    """)
    
    with st.form("stress_checkin"):
        stress_level = st.slider(
            "Current Stress Level",
            0, 10, 5,
            help="0 = Completely relaxed, 10 = Extremely stressed"
        )
        
        physical_symptoms = st.multiselect(
            "Physical Symptoms",
            [
                "Tension headache",
                "Muscle tension",
                "Rapid heartbeat",
                "Shallow breathing",
                "Stomach discomfort",
                "Fatigue",
                "Sweating",
                "Other"
            ]
        )
        
        if "Other" in physical_symptoms:
            other_symptoms = st.text_input("Specify other symptoms:")
        
        triggers = st.multiselect(
            "Current Stress Triggers",
            [
                "Work/School",
                "Relationships",
                "Health",
                "Financial",
                "Time pressure",
                "Social situations",
                "Future uncertainty",
                "Other"
            ]
        )
        
        if "Other" in triggers:
            other_triggers = st.text_input("Specify other triggers:")
        
        coping_methods = st.multiselect(
            "What helps you cope?",
            [
                "Deep breathing",
                "Exercise",
                "Talking to someone",
                "Time in nature",
                "Meditation",
                "Music",
                "Creative activities",
                "Other"
            ]
        )
        
        if "Other" in coping_methods:
            other_methods = st.text_input("Specify other coping methods:")
        
        notes = st.text_area(
            "Additional Notes",
            placeholder="Any other thoughts or observations..."
        )
        
        submitted = st.form_submit_button("Save Check-In", use_container_width=True)
        if submitted:
            # Format data for logging
            checkin_notes = f"""
Stress Level: {stress_level}/10
Physical Symptoms: {', '.join(physical_symptoms) if physical_symptoms else 'None'}
Triggers: {', '.join(triggers) if triggers else 'None'}
Coping Methods: {', '.join(coping_methods) if coping_methods else 'None'}
Notes: {notes if notes else 'None'}
            """.strip()
            
            # Store for potential discussion
            st.session_state.stress_checkin_data_for_discussion = {
                "level": stress_level,
                "symptoms": physical_symptoms,
                "triggers": triggers,
                "coping": coping_methods,
                "notes": notes
            }

            handle_exercise_logging(
                "Stress Check-In",
                stress_level,
                checkin_notes
            )

def render_exposure_planner():
    """Render the Graded Exposure Planner exercise."""
    st.markdown("""
        Break down challenging tasks into smaller, manageable steps to reduce
        anxiety and build confidence gradually.
    """)
    
    if "exposure_task" not in st.session_state:
        st.session_state.exposure_task = ""
        st.session_state.exposure_steps = []
    
    # Task identification
    task = st.text_input(
        "What task or situation would you like to work on?",
        value=st.session_state.exposure_task
    )
    
    if task and task != st.session_state.exposure_task:
        st.session_state.exposure_task = task
        st.rerun()
    
    if task:
        st.markdown("### Break it down")
        st.markdown("""
            Create 3-5 steps, starting with the least anxiety-provoking
            and gradually working up to your goal.
        """)
        
        # Add new step
        with st.form("add_step"):
            col1, col2 = st.columns([3, 1])
            with col1:
                new_step = st.text_input("Step description")
            with col2:
                anxiety = st.number_input(
                    "Anxiety (0-10)",
                    min_value=0,
                    max_value=10,
                    value=5
                )
            
            if st.form_submit_button("Add Step"):
                st.session_state.exposure_steps.append({
                    "description": new_step,
                    "anxiety": anxiety,
                    "completed": False
                })
                st.rerun()
        
        # Show steps
        if st.session_state.exposure_steps:
            st.markdown("### Your Steps")
            for i, step in enumerate(st.session_state.exposure_steps):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(
                        f"{'✅' if step['completed'] else '⬜'} "
                        f"{i+1}. {step['description']}"
                    )
                with col2:
                    st.markdown(f"Anxiety: {step['anxiety']}/10")
                with col3:
                    if not step['completed'] and (
                        i == 0 or 
                        (i > 0 and st.session_state.exposure_steps[i-1]['completed'])
                    ):
                        if st.button("Complete", key=f"complete_{i}"):
                            st.session_state.exposure_steps[i]['completed'] = True
                            st.rerun()
            
            # Progress visualization
            completed = sum(1 for step in st.session_state.exposure_steps if step['completed'])
            total = len(st.session_state.exposure_steps)
            st.progress(completed / total)
            st.markdown(f"Progress: {completed}/{total} steps completed")
            
            if completed == total:
                st.success("🎉 Congratulations! You've completed all steps!")

            # Add "Clear Only Steps" button if there are steps
            if st.button(_("Clear Only Steps"), key="clear_exposure_steps_only"):
                st.session_state.exposure_steps = []
                # Also reset discussion context related to steps if it exists
                if "exposure_plan_data_for_discussion" in st.session_state:
                    st.session_state.exposure_plan_data_for_discussion["steps"] = "No steps defined"
                st.rerun()
        
        # Clear plan button (Task & Steps)
        if st.button(_("Clear Plan (Task & Steps)"), key="clear_exposure_plan"):
            st.session_state.exposure_task = ""
            st.session_state.exposure_steps = []
            st.rerun()
    else:
        st.info("""
            Examples:
            - Making a phone call
            - Going to a social event
            - Speaking in public
            - Trying something new
        """)

# Helper function to delete a task from the Eisenhower Matrix
def _delete_matrix_task(quadrant_key, task_index):
    if quadrant_key in st.session_state.matrix_items and 0 <= task_index < len(st.session_state.matrix_items[quadrant_key]):
        del st.session_state.matrix_items[quadrant_key][task_index]
        st.rerun()

# Helper function to render tasks within a quadrant
def _render_quadrant_content(title, quadrant_key, quadrant_css_class):
    st.markdown(f'<div class="matrix-title">{title}</div>', unsafe_allow_html=True)
    
    tasks = st.session_state.matrix_items[quadrant_key]
    if not tasks:
        st.caption(_("No tasks here yet."))
    else:
        for idx, task in enumerate(tasks):
            task_key_suffix = f"matrix_task_{quadrant_key}_{idx}" # Used for button key uniqueness
            del_button_key = f"del_matrix_task_{quadrant_key}_{idx}"
        
            # Use columns for task text and delete button side-by-side
            col_task_text, col_task_button = st.columns([0.85, 0.15])
            with col_task_text:
                st.markdown(task) # Render task text directly
            with col_task_button:
                if st.button("🗑️", key=del_button_key, help=_("Delete task")):
                    _delete_matrix_task(quadrant_key, idx)
                    # No st.rerun() needed here as _delete_matrix_task does it
    
    # Removed the explicit task-item div and per-task st.container

def render_eisenhower_matrix():
    """Render an interactive Eisenhower Matrix."""
    st.subheader("🎯 Eisenhower Matrix")

    # Eisenhower Matrix CSS is now in style.css

    # Task input form
    with st.form(key="matrix_task_form", clear_on_submit=True):
        st.markdown("<h5>➕ Add a New Task</h5>", unsafe_allow_html=True)
        task_description = st.text_input(_("Task Description"), placeholder=_("e.g., Finish report"))
        col1_form, col2_form = st.columns(2)
        with col1_form:
            is_urgent = st.checkbox(_("Urgent"))
        with col2_form:
            is_important = st.checkbox(_("Important"))
        submitted = st.form_submit_button(_("Add Task"))

        if submitted and task_description:
            if is_important and is_urgent:
                st.session_state.matrix_items['important_urgent'].append(task_description)
            elif is_important and not is_urgent:
                st.session_state.matrix_items['important_not_urgent'].append(task_description)
            elif not is_important and is_urgent:
                st.session_state.matrix_items['not_important_urgent'].append(task_description)
            else:
                st.session_state.matrix_items['not_important_not_urgent'].append(task_description)
            st.success(_(f"Task '{task_description}' added!"))
            # No st.rerun() needed here if form submission naturally causes it, 
            # or if adding task modifies state observed by elements below.
            # However, to be safe and ensure UI update for the new task:
            st.rerun() 
        elif submitted and not task_description:
            st.warning(_("Please enter a task description."))

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div class="matrix-label">Importance ↑ | Urgency →</div>', unsafe_allow_html=True)
    
    # Matrix rendering using Streamlit columns within the CSS grid wrapper
    st.markdown('<div class="matrix-content-wrapper">', unsafe_allow_html=True)
    
    # Row 1
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="matrix-quadrant do-first-quadrant">', unsafe_allow_html=True)
        _render_quadrant_content(_("🔥 Do First"), 'important_urgent', 'do-first-quadrant')
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="matrix-quadrant schedule-quadrant">', unsafe_allow_html=True)
        _render_quadrant_content(_("📅 Schedule"), 'important_not_urgent', 'schedule-quadrant')
        st.markdown('</div>', unsafe_allow_html=True)
        
    # Row 2
    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="matrix-quadrant delegate-quadrant">', unsafe_allow_html=True)
        _render_quadrant_content(_("👤 Delegate"), 'not_important_urgent', 'delegate-quadrant')
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="matrix-quadrant delete-quadrant">', unsafe_allow_html=True)
        _render_quadrant_content(_("🗑️ Eliminate"), 'not_important_not_urgent', 'delete-quadrant') # Changed title from "Delete" to "Eliminate" to avoid confusion with delete button
        st.markdown('</div>', unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True) # Close matrix-content-wrapper

    st.markdown("<hr style='margin-top: 30px;'>", unsafe_allow_html=True)

    # Button to clear all matrix tasks
    if any(st.session_state.matrix_items.values()): # Show button only if there are tasks
        if st.button(_("Clear All Matrix Tasks"), key="clear_all_matrix_tasks"):
            st.session_state.matrix_items = {
                'important_urgent': [],
                'important_not_urgent': [],
                'not_important_urgent': [],
                'not_important_not_urgent': []
            }
            # Clear discussion context if it exists
            if "matrix_content_for_ai" in st.session_state: # Assuming this var name or similar for context
                del st.session_state.matrix_content_for_ai # Or reset it appropriately
            st.success(_("All tasks cleared from the matrix!"))
            st.rerun()

    # Button to discuss matrix content
    matrix_discuss_button_text = _("Discuss Matrix Tasks with AI")
    
    if st.button(matrix_discuss_button_text, key="discuss_matrix_content"):
        matrix_content_for_ai = "Here are my current Eisenhower Matrix tasks:\\n"
        for quadrant, tasks in st.session_state.matrix_items.items():
            quadrant_title = quadrant.replace('_', ' ').replace('important', 'Important').replace('urgent', 'Urgent').replace('not', 'Not').capitalize()
            matrix_content_for_ai += f"\\n**{quadrant_title}:**\\n"
            if tasks:
                for task_item in tasks:
                    matrix_content_for_ai += f"- {task_item}\\n"
            else:
                matrix_content_for_ai += "- (No tasks)\\n"
        
        final_matrix_prompt = _("I'd like to discuss my Eisenhower Matrix tasks. Here's what I have:\\n{matrix_data}\\nWhat are your suggestions or insights?").format(matrix_data=matrix_content_for_ai)

        st.session_state.active_tab = "AI Chat"
        if "messages" not in st.session_state:
            st.session_state.messages = []
        st.session_state.messages.append({"role": "user", "content": final_matrix_prompt})
        st.rerun()

# Global definition of exercise data
EXERCISES_DATA = {
    _("Box Breathing"): {"icon": "🧘‍♀️", "desc": _("A simple technique for calming your nerves."), "func": render_box_breathing, "video_url": "https://www.youtube.com/watch?v=tEmt1Znux58"},
    _("5-4-3-2-1 Grounding"): {"icon": "🌳", "desc": _("Bring yourself to the present moment."), "func": render_grounding, "video_url": "https://www.youtube.com/watch?v=30VMIEmA114"},
    _("Progressive Muscle Relaxation"): {"icon": "💪", "desc": _("Relax your body, one muscle group at a time."), "func": render_pmr, "video_url": "https://www.youtube.com/watch?v=1nZEdqcGVzo"},
    _("Likert Stress Check-in"): {"icon": "📊", "desc": _("Assess your current stress level."), "func": render_stress_checkin},
    _("Graded Exposure Planner"): {"icon": "🪜", "desc": _("Plan steps to face your fears."), "func": render_exposure_planner, "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}, # Note: Rickroll link, user might want to change this
    _("Eisenhower Matrix"): {"icon": "📅", "desc": _("Prioritize tasks effectively."), "func": render_eisenhower_matrix}
}

# Initialize OpenAI client
client = OpenAI()

# Function to handle PDF uploads (if it wasn't moved already)
# Assuming handle_pdf_upload, detect_critical_situation are defined before this point if not moved

def handle_pdf_upload(uploaded_file):
    """Handle PDF file upload with text and image extraction"""
    # Create a temporary file to save the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        pdf_path = tmp_file.name

    try:
        # Extract text from PDF
        text_content = []
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text_content.append(page.extract_text())

        # Convert PDF pages to images and extract text using OCR
        images = convert_from_path(pdf_path)
        ocr_text = []
        for image in images:
            text = pytesseract.image_to_string(image)
            ocr_text.append(text)

        # Combine extracted text
        all_text = "\\n".join(text_content + ocr_text)

        # Save the combined text analysis
        save_path = DATA_DIR / "data_library" / "subjects" / uploaded_file.name
        analysis_path = save_path.with_suffix('.analysis.txt')
        
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with open(analysis_path, "w", encoding='utf-8') as f:
            f.write(all_text)

        st.success(f"Successfully processed {uploaded_file.name}")
        
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        logger.error(f"PDF processing error: {str(e)}")
    
    finally:
        # Clean up temporary file
        os.unlink(pdf_path)

def detect_critical_situation(message: str) -> bool:
    """Detect if the user's message indicates a critical mental health situation."""
    critical_phrases = [
        "want to die", "kill myself", "suicide", "end my life", "don't want to live",
        "want to hurt myself", "self harm", "self-harm", "cut myself",
        "no reason to live", "better off dead", "can't take it anymore"
    ]
    return any(phrase in message.lower() for phrase in critical_phrases)

def initialize_session_state():
    """Initializes all necessary session state variables if they don't exist."""
    if "ui_lang" not in st.session_state:
        st.session_state.ui_lang = CONFIG["DEFAULT_LANGUAGE"]
    
    # Initialize gettext_translations based on ui_lang
    if "gettext_translations" not in st.session_state:
        lang_to_load = st.session_state.get("ui_lang", CONFIG["DEFAULT_LANGUAGE"])
        fallback_lang = CONFIG["DEFAULT_LANGUAGE"] if "en" not in LANGUAGES else "en"
        st.session_state.gettext_translations = LANGUAGES.get(lang_to_load, LANGUAGES.get(fallback_lang))

    if "profile_data" not in st.session_state:
        st.session_state.profile_data = {} 
        if "onboarding_completed" not in st.session_state.profile_data:
             st.session_state.profile_data["onboarding_completed"] = False
        if "country" not in st.session_state.profile_data: 
             st.session_state.profile_data["country"] = ""

    if "messages" not in st.session_state:
        # Enhanced system prompt
        system_prompt_content = _("""You are **MindShield**, an empathetic AI mental-health assistant.  
Your purpose is to offer warm, evidence-based support while guiding users to this app's built-in tools.  
Always keep the user's well-being, privacy, and safety at the centre of every reply.

─────────────────────────────────
THERAPEUTIC IDENTITY & STANCE
─────────────────────────────────
• Show unconditional positive regard, empathy, and cultural humility.  
• Draw flexibly from CBT, ACT, psychodynamic insight, person-centred and solution-focused principles, plus trauma-informed care.  
• Use plain language; explain any jargon only if the user wants it.  
• Mirror the user's language (English, Arabic, French) whenever feasible, staying consistent within a single reply.

─────────────────────────────────
STANDARD SESSION FLOW  (adapt as needed)
─────────────────────────────────
1. **Contract / Goal-setting** – agree on today's focus.  
2. **Explore & Reflect** – active listening, validation, clarifying questions.  
3. **Insight / Conceptualise** – link thoughts, feelings, patterns, values.  
4. **Intervene** – introduce one relevant tool or skill with permission.  
5. **Summarise & Next Steps** – recap, optional homework, supportive close.

─────────────────────────────────
EVIDENCE-BASED TOOLBOX  (offer only when relevant)
─────────────────────────────────
• Eisenhower Matrix (task triage) • 5-4-3-2-1 Grounding  
• Box Breathing • Progressive Muscle Relaxation • Likert check-ins  
• SMART or WOOP goals • Behavioural-activation planner  
• Cognitive-restructuring worksheet • Values clarification / card sort  
• Journaling, gratitude, self-compassion exercises  
• **Time-Management Suite** (see "Time-Management Guidance" below)  
  – GTD workflow – SMART goal planner – DQD interruption handling  
  – Inbox-Zero / R.A.S.A.T. email method – Parkinson / Pareto / Illich blocks

─────────────────────────────────
CRISIS & RISK PROTOCOL  (override all else)
─────────────────────────────────
If the user expresses self-harm, suicidal intent, intent to harm others, or is in immediate danger:

1. **Stay calm & empathise.**  
2. **Assess risk** (plan, means, time-frame) with caring, direct questions.  
3. **Encourage immediate help:**  
   • Dial **999** (police) or **998** (ambulance) in the UAE for any life-threatening emergency.  
   • Call the UAE's 24/7 mental-support line **800 HOPE (800 4673)** or, in Dubai, **800 111**.  
   • If outside the UAE, contact local emergency services or a trusted crisis line.  
4. Offer grounding while help is sought. **Never** provide lethal-means instructions.  
5. Offer professional follow-up:  
   "I can help you book a therapist session now. [BUTTON: Book a Therapist Session: action_navigate_therapist_booking]  
   (You can also type `/book` if that's easier.)"  
   Resume other support only after safety is sufficiently addressed.

─────────────────────────────────
BOUNDARIES & ETHICS
─────────────────────────────────
• You are an AI assistant, **not** a licensed clinician; no diagnosis, prescriptions, legal or financial advice.  
• Maintain professionalism; refuse disallowed or unethical requests.  
• If asked about confidentiality, explain that conversations are stored but not shared with third parties.  
• Keep replies ≤ 4 sentences per paragraph; adapt tone to the user; minimal emojis unless the user uses them first.

─────────────────────────────────
NEW CAPABILITIES  – HOW TO INVOKE APP FEATURES
─────────────────────────────────
1. **Suggesting In-App Tools or Professional Support**  
   Embed a navigation button when a feature clearly fits the user's need, or if they might benefit from booking a session.  
   Format: `[BUTTON: Button Label: action_code]`  
   Action codes:  
   • `action_navigate_eisenhower` Eisenhower Matrix  
   • `action_navigate_box_breathing` Box Breathing  
   • `action_navigate_grounding` 5-4-3-2-1 Grounding  
   • `action_navigate_pmr` Progressive Muscle Relaxation  
   • `action_navigate_stress_checkin` Stress Check-in  
   • `action_navigate_exposure_planner` Graded Exposure Planner  
   • `action_navigate_therapist_booking` Book Therapist Session  

   *Slash-command compatibility*  
   • `/book`  → treat as `action_navigate_therapist_booking`  
   • `/exercise` → open the Exercise tab (user chooses activity)

2. **Accessing the App's File System**  
   Include one of these commands when background material will help:  
   • `[APP_REQUEST: LIST_FILES folder_type="subjects|feedback"]`  
   • `[APP_REQUEST: READ_FILE folder_type="subjects|feedback" filename="<name>"]`  
   • `[APP_REQUEST: SEARCH_FILES folder_type="subjects|feedback" keywords="<kw1,kw2>"]`  
   After issuing a request, **pause and wait** for the app's reply before continuing.

─────────────────────────────────
DATA-LIBRARY INSIGHTS  (ALWAYS INTEGRATED IMPLICITLY)
─────────────────────────────────
• **French emotional-distress flow**  
  – Empathise in French ("Je suis désolé…")  
  – Ask what thoughts or sensations worry them.  
  – Offer a micro-coping tool (three deep breaths / short walk).  
  – Suggest `/exercise` for a calming activity.  
  – Prompt self-kindness ("Comment pourrais-tu te montrer un peu de gentillesse aujourd'hui ?").

• **Limited-time situations**  
  – Validate lack of time and suggest **micro-strategies** (e.g., three deep breaths on the spot).  
  – Ask how they feel afterwards; celebrate small relief.

• **User feels better**  
  – Reinforce improvement and remind them the quick technique is repeatable.  
  – Offer next-step choices:  
    1. set relaxation reminder 2. time-management help 3. explore remaining concerns.

• **Time-management coaching flow**  
  – Confirm interest ("Je veux 2" etc.).  
  – Ask for a to-do list ordered from simplest → most complex, including deadlines / importance.  
  – Guide prioritisation and delegation once list is provided.  
  – Respect postponement; invite return anytime.

─────────────────────────────────
TIME-MANAGEMENT GUIDANCE  (core concepts from library)
─────────────────────────────────
• **Psychological Foundations** – Explain the stress cost of multitasking and urgency culture; highlight attention fragmentation.  
• **Key Laws:** Parkinson (tasks expand), Murphy (build 1.5–3× buffer), Pareto 80/20, Illich (≤ 90-min focus blocks), Laborit (do hard tasks first), Carlson (protect focus from interruptions).  
• **Eisenhower Matrix – advanced tips**  
  – Quadrant UI (urgent + important): schedule buffers for the unexpected.  
  – Quadrant uI (non-urgent + important): block non-negotiable proactive time.  
  – Quadrant Ui (urgent + not-important): negotiate, delegate, or batch.  
  – Quadrant ui (non-urgent + not-important): minimise or eliminate.  
• **DQD Interruption Method** – *Dissuade • Question • Decide* for each interruption (< 2 min → do; else schedule / delegate / decline).  
• **SMART Goal Framing** – Specific, Measurable, Ambitious, Realistic, Time-bound.  
• **GTD Workflow** – Capture → Clarify → Organise → Review weekly.  
• **Inbox-Zero / R.A.S.A.T.** – Reply • Add to list • Suppress • Archive • Transmit/delegate.  
• **Planning Heuristics** – Block big rocks first; reserve ~20 % buffer; build 15-min transition gaps between meetings.  
When users seek time-management help, introduce one relevant principle at a time, keep examples concrete, and tie back to their stated goals.

─────────────────────────────────
EXAMPLES  (do not include in normal replies)
─────────────────────────────────
• "I'm drowning in tasks."  
  → "The Eisenhower Matrix can help you prioritise. [BUTTON: Open Eisenhower Matrix: action_navigate_eisenhower]"

• "I'm shaky and can't calm down."  
  → "Let's try Box Breathing together. [BUTTON: Try Box Breathing: action_navigate_box_breathing]"

• "Je n'ai pas le temps de faire l'exercice."  
  → "Je comprends que ton temps est limité. Essayons trois grandes respirations ensemble—où que tu sois—et dis-moi comment tu te sens après."

• "Do you have material on social-anxiety coping?"  
  → "I can search our library. [APP_REQUEST: SEARCH_FILES folder_type="subjects" keywords="social anxiety,coping"]"

• "I feel overwhelmed and don't know who to turn to."  
  → "It sounds like you're going through a lot. Speaking with a professional could help. [BUTTON: Book a Therapist Session: action_navigate_therapist_booking]"

• User: "I want to die."  
  → AI: *follows Crisis & Risk Protocol above.*

─────────────────────────────────
MINDSHIELD TIME-MANAGEMENT COACHING  •  EXTENDED FLOW
─────────────────────────────────
1. **Initial Inventory** – ask in this order:  
   • A list of current tasks with their deadlines (e.g., "buy groceries by tomorrow", "write report this weekend") or any regularly recurring duties.  
   • A list of objectives the user hasn't yet achieved but wants to (e.g., "learn English", "watch a film with my partner").  

2. **Choose an Approach** – offer two clear options:  
   a. **Done-For-You Triage** – MindShield builds the Eisenhower Matrix for today and returns an action plan (utility mode).  
   b. **Skill-Building Guidance** – coach the user step-by-step to organise their whole week (learning mode).  

3. **Routine Mapping** – invite the user to sketch a "typical day" agenda (wake-up to bedtime).  
""")
