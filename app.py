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

# --- MindShield Retrieval & Chat Engine Setup (added) ---
from mindshield_core.retrieval import WindowRetriever
from mindshield_core.chat_engine import ChatEngine

# Create a very small in-memory index so the retriever works even when
# no knowledge sources have been uploaded yet. Each record only needs an
# 'id' and 'text' field for WindowRetriever; keep it empty for now.
_RETRIEVAL_INDEX: list = []  # Will be populated later from uploaded files.

# Instantiate global retriever and chat engine once so they can be reused
# across Streamlit reruns (they will be stored in st.session_state later if desired).
RETRIEVER = WindowRetriever(_RETRIEVAL_INDEX)
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
logger = logging.getLogger(__name__)
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
    # Unique session identifier for logging / retrieval
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

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

4. **Task Refinement** – suggest turning the raw task list into a structured to-do list and re-ordering by priority.  

5. **Eisenhower Exercise & Discussion** –  
   • Walk through creating their matrix (or present the matrix if option a).  
   • Explain the difference between *important* and *urgent*; ask clarifying questions and gently correct misunderstandings.  
   • Spot and label *time-wasters* (perfectionism, phone scrolling, talkative colleague, etc.). If unclear, ask probing questions so the user identifies them; place these in "neither urgent nor important" and, where useful, introduce the **DQD Interruption Method**.  
   • Show how procrastination can push items into the "urgent & important" quadrant.  

6. **Agenda Optimisation** – help rebuild the user's schedule so daily routines align with their goals and values.  

7. **Seven-Day Practice** – invite the user to record their real agenda each day for a week and review progress together at the end (handled in chat; no in-app tracker yet).  

─────────────────────────────────
FINAL REMINDERS
─────────────────────────────────
• Empathy, clarity, and user safety come first.  
• Offer buttons, slash-commands, or file requests only when they add clear value.  
• Keep every response user-centred, actionable, culturally aware, and within these ethical bounds.""")
        st.session_state.messages = [{"role": "system", "content": system_prompt_content}]
        
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Home"
        
    if "show_therapist_button" not in st.session_state:
        st.session_state.show_therapist_button = False
        
    if "matrix_items" not in st.session_state:
        st.session_state.matrix_items = {
            'important_urgent': [],
            'important_not_urgent': [],
            'not_important_urgent': [],
            'not_important_not_urgent': []
        }
        
    if "active_exercise" not in st.session_state:
        st.session_state.active_exercise = None
        
    # For "Discuss Exercise Content with AI" feature, initialize data stores if not present
    if "grounding_items" not in st.session_state: 
        st.session_state.grounding_items = {
            "see": [], "feel": [], "hear": [], "smell": [], "taste": []
        }
    if "exposure_task" not in st.session_state: 
        st.session_state.exposure_task = ""
    if "exposure_steps" not in st.session_state: 
        st.session_state.exposure_steps = []
    if "stress_checkin_data_for_discussion" not in st.session_state: 
        st.session_state.stress_checkin_data_for_discussion = {}

def handle_ai_suggested_action(action_code: str):
    """Handles navigation or other actions suggested by the AI."""
    logger.info(f"Handling AI suggested action: {action_code}")
    if action_code == "action_navigate_eisenhower":
        st.session_state.active_tab = "Exercises"
        set_active_exercise(_("Eisenhower Matrix")) # Ensure this name matches EXERCISES_DATA key
    elif action_code == "action_navigate_box_breathing":
        st.session_state.active_tab = "Exercises"
        set_active_exercise(_("Box Breathing"))
    elif action_code == "action_navigate_grounding":
        st.session_state.active_tab = "Exercises"
        set_active_exercise(_("5-4-3-2-1 Grounding"))
    elif action_code == "action_navigate_pmr":
        st.session_state.active_tab = "Exercises"
        set_active_exercise(_("Progressive Muscle Relaxation"))
    elif action_code == "action_navigate_stress_checkin":
        st.session_state.active_tab = "Exercises"
        set_active_exercise(_("Likert Stress Check-in"))
    elif action_code == "action_navigate_exposure_planner":
        st.session_state.active_tab = "Exercises"
        set_active_exercise(_("Graded Exposure Planner"))
    elif action_code == "action_navigate_therapist_booking":
        st.session_state.active_tab = "Therapist"
    # Add more actions here if needed
    else:
        logger.warning(f"Unknown AI action code: {action_code}")
        return # Do nothing if action code is unknown

    st.rerun()

# --- AI File Operation Helper Functions ---
MAX_FILE_READ_SIZE = 10000 # Max characters to read from a file for AI
MAX_SEARCH_RESULTS_TOTAL_SIZE = 2000 # Max total characters for all search snippets
MAX_SNIPPET_SIZE = 500 # Max characters for a single search snippet (search looks around keyword)

def get_ai_accessible_path(folder_type: str, filename: Optional[str] = None) -> Optional[Path]:
    """Returns the Path object for AI accessible folders/files, or None if invalid."""
    base_path = None
    if folder_type == "subjects":
        base_path = DATA_LIBRARY_DIR / "subjects"
        base_path.mkdir(parents=True, exist_ok=True) # Ensure it exists
    elif folder_type == "feedback":
        base_path = FEEDBACK_DIR
    else:
        logger.warning(f"Invalid folder_type specified for AI operation: {folder_type}")
        return None

    if filename:
        # Sanitize filename to prevent path traversal
        if ".." in filename or "/" in filename or "\\\\" in filename or filename.startswith(".") :
            logger.warning(f"Potentially unsafe filename specified: {filename}")
            return None
        return base_path / Path(filename).name # Use Path(filename).name to further sanitize
    return base_path

def list_files_for_ai(folder_type: str) -> str:
    logger.info(f"AI requested to list files in folder_type: {folder_type}")
    path = get_ai_accessible_path(folder_type)
    if not path or not path.is_dir():
        return _("Application Error: Could not access the specified folder: {folder_type}.").format(folder_type=folder_type)
    
    try:
        files = [f.name for f in path.iterdir() if f.is_file()]
        if not files:
            return _("No files found in the '{folder_type}' folder.").format(folder_type=folder_type)
        return _("Files available in '{folder_type}':\\n - ").format(folder_type=folder_type) + "\\n - ".join(files)
    except Exception as e:
        logger.error(f"Error listing files for AI in {folder_type}: {e}")
        return _("Application Error: Could not list files.")

def read_file_for_ai(folder_type: str, filename: str) -> str:
    logger.info(f"AI requested to read file: {filename} from folder_type: {folder_type}")
    file_path = get_ai_accessible_path(folder_type, filename)
    if not file_path or not file_path.is_file():
        return _("Application Error: File '{filename}' not found or path is invalid in {folder_type}.").format(filename=filename, folder_type=folder_type)
    
    try:
        content = file_path.read_text(encoding='utf-8', errors='replace')[:MAX_FILE_READ_SIZE]
        return _("Content of '{filename}' (first {MAX_FILE_READ_SIZE} chars):\\n{content}").format(
            filename=filename, MAX_FILE_READ_SIZE=MAX_FILE_READ_SIZE, content=content
        )
    except Exception as e:
        logger.error(f"Error reading file {filename} for AI: {e}")
        return _("Application Error: Could not read file '{filename}'.").format(filename=filename)

def search_files_for_ai(folder_type: str, keywords_str: str) -> str:
    logger.info(f"AI requested to search in {folder_type} for keywords: {keywords_str}")
    base_path = get_ai_accessible_path(folder_type)
    if not base_path or not base_path.is_dir():
        return _("Application Error: Could not access the specified folder '{folder_type}' for search.").format(folder_type=folder_type)

    keywords = [k.strip().lower() for k in keywords_str.split(',') if k.strip()]
    if not keywords:
        return _("Application Error: No keywords provided for search.")

    found_snippets = []
    total_chars_retrieved = 0

    try:
        for item in base_path.iterdir():
            if item.is_file():
                if total_chars_retrieved >= MAX_SEARCH_RESULTS_TOTAL_SIZE:
                    found_snippets.append(_("... [Search results truncated due to size limit] ..."))
                    break 
                try:
                    content = item.read_text(encoding='utf-8', errors='replace')
                    content_lower = content.lower()
                    for keyword in keywords:
                        if total_chars_retrieved >= MAX_SEARCH_RESULTS_TOTAL_SIZE: break
                        
                        idx = content_lower.find(keyword)
                        while idx != -1:
                            if total_chars_retrieved >= MAX_SEARCH_RESULTS_TOTAL_SIZE: break
                            
                            # Calculate snippet boundaries
                            half_snippet = (MAX_SNIPPET_SIZE - len(keyword)) // 2
                            snippet_start = max(0, idx - half_snippet)
                            snippet_end = min(len(content), idx + len(keyword) + half_snippet)
                            snippet = content[snippet_start:snippet_end]

                            # Add ellipsis if snippet is cut
                            prefix = "..." if snippet_start > 0 else ""
                            suffix = "..." if snippet_end < len(content) else ""
                            
                            formatted_snippet = f"{prefix}{snippet}{suffix}"

                            if total_chars_retrieved + len(formatted_snippet) > MAX_SEARCH_RESULTS_TOTAL_SIZE and found_snippets:
                                found_snippets.append(_("... [Search results truncated due to size limit] ..."))
                                total_chars_retrieved = MAX_SEARCH_RESULTS_TOTAL_SIZE # Mark as full
                                break 
                            
                            found_snippets.append(f"Found '{keyword}' in {item.name}: {formatted_snippet}")
                            total_chars_retrieved += len(formatted_snippet)
                            
                            next_search_start = idx + len(keyword)
                            if next_search_start >= len(content_lower): break
                            idx = content_lower.find(keyword, next_search_start)
                        if total_chars_retrieved >= MAX_SEARCH_RESULTS_TOTAL_SIZE: break 
                except Exception as e:
                    logger.warning(f"Could not read or search file {item.name} for AI: {e}")
                    if total_chars_retrieved < MAX_SEARCH_RESULTS_TOTAL_SIZE:
                         found_snippets.append(_("Application Error: Error processing file: {filename}").format(filename=item.name))
                         total_chars_retrieved += len("Application Error: Error processing file: ")
            if total_chars_retrieved >= MAX_SEARCH_RESULTS_TOTAL_SIZE: break

        if not found_snippets:
            return _("No results found for keywords: '{keywords_str}' in {folder_type}.").format(keywords_str=keywords_str, folder_type=folder_type)
        
        return _("Search results for '{keywords_str}' in {folder_type}:\\n\\n").format(keywords_str=keywords_str, folder_type=folder_type) + "\\n---\\n".join(found_snippets)

    except Exception as e:
        logger.error(f"Error searching files for AI in {folder_type} with keywords {keywords_str}: {e}")
        return _("Application Error: Could not perform search.")

def handle_ai_file_operation(command: str, params_str: str) -> str:
    """Handles file operations requested by the AI by parsing params_str."""
    logger.info(f"Handling AI file operation command: {command}, params_str: {params_str}")
    
    params = {}
    folder_type_match = re.search(r'folder_type\s*=\s*"(subjects|feedback)"', params_str)
    if folder_type_match:
        params["folder_type"] = folder_type_match.group(1)
    else:
        return _("Application Error: 'folder_type' (subjects or feedback) missing or invalid in AI request.")

    filename_match = re.search(r'filename\s*=\s*"([^"]*)"', params_str)
    if filename_match:
        params["filename"] = filename_match.group(1)

    keywords_match = re.search(r'keywords\s*=\s*"([^"]*)"', params_str)
    if keywords_match:
        params["keywords"] = keywords_match.group(1)

    folder_type = params.get("folder_type") # Already checked it's valid

    if command == "LIST_FILES":
        return list_files_for_ai(folder_type)
    elif command == "READ_FILE":
        filename = params.get("filename")
        if not filename:
            return _("Application Error: 'filename' missing for READ_FILE operation in AI request.")
        return read_file_for_ai(folder_type, filename)
    elif command == "SEARCH_FILES":
        keywords = params.get("keywords")
        if not keywords:
            return _("Application Error: 'keywords' missing for SEARCH_FILES operation in AI request.")
        return search_files_for_ai(folder_type, keywords)
    else:
        logger.warning(f"Unknown AI file operation command received: {command}")
        return _("Application Error: Unknown AI file operation command.")
# --- End AI File Operation Helper Functions ---

# --- Utility: tail log file for live display ---
def _get_recent_logs(path: str = "logs/app.log", max_lines: int = 300) -> str:
    """Return up to *max_lines* of the end of the given log file (safe)."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()[-max_lines:]
            return "".join(lines) if lines else "(log file is empty)"
    except FileNotFoundError:
        return "(log file not found)"
    except Exception as e:
        return f"(error reading log: {e})"

# ------------------------------------------------------------------
# CHAT TAB – now with side-by-side live logs
# ------------------------------------------------------------------

def render_chat_tab():
    st.header(_("💭 Chat with AI Assistant"))

    # Create two columns: 2/3 for chat, 1/3 for logs
    col_chat, col_logs = st.columns([2, 1])

    # ---------------- Left: Chat UI ----------------
    with col_chat:
        _render_chat_interface()

    # ---------------- Right: Live Logs -------------
    with col_logs:
        st.subheader("🪵 Live Logs")
        # Auto-refresh every 5 s while the user is on this tab
        from streamlit_autorefresh import st_autorefresh  # lightweight dep (<5 kB)
        st_autorefresh(interval=5000, key="log_autorefresh")
        st.code(_get_recent_logs(), language="text")

# ------------------------------------------------------------------
# Original chat logic moved into helper to keep render_chat_tab tidy
# ------------------------------------------------------------------

def _render_chat_interface():
    """All existing chat-handling code extracted here (verbatim)."""
    api_key = os.getenv('OPENAI_API_KEY')
    logger.debug(f"OpenAI API Key status: {'Present' if api_key else 'Missing'}")
    if not api_key:
        st.error(_("OpenAI API Key is missing. Please check your .env file."))
        return

    with st.sidebar:
        is_trainer = st.checkbox(_("👩‍⚕️ Trainer Mode"), help=_("Enable this mode to correct AI responses"))

    button_regex = r"\[BUTTON: ([^:]+): ([a-zA-Z0-9_]+)\]"
    app_request_regex = r"\[APP_REQUEST:\s*(LIST_FILES|READ_FILE|SEARCH_FILES)\s*(.*?)\]"

    # Display chat messages
    for message_idx, message in enumerate(st.session_state.messages):
        if message["role"] == "system":
            continue  # Don't display system messages

        with st.chat_message(message["role"]):
            if message["role"] == "tool":  # Handle tool responses specifically
                st.markdown(f"**Application Response (File Operation):**\n```\n{message['content']}\n```")
            else:  # User or Assistant messages
                content_parts = re.split(button_regex, message["content"])
                # Example: "Text [BUTTON:L1:A1] More [BUTTON:L2:A2] End"
                # re.split -> ['Text ', 'L1', 'A1', ' More ', 'L2', 'A2', ' End']

                part_processing_idx = 0
                while part_processing_idx < len(content_parts):
                    # Part 1: Text segment (always present, might be empty)
                    text_segment = content_parts[part_processing_idx]
                    if text_segment:
                        st.markdown(text_segment, unsafe_allow_html=True)
                    part_processing_idx += 1

                    # Part 2 & 3: Button Label and Action Code (if they exist as a pair)
                    if part_processing_idx < len(content_parts):  # We have a potential label
                        button_label = content_parts[part_processing_idx].strip()
                        part_processing_idx += 1
                        if part_processing_idx < len(content_parts):  # We have an action code
                            action_code = content_parts[part_processing_idx].strip()
                            part_processing_idx += 1  # Consumed action code

                            # Create the button
                            button_key = (
                                f"ai_action_btn_{message_idx}_{action_code.replace(' ', '_')}_{part_processing_idx}"
                            )  # Unique key, sanitize action_code for key
                            st.button(
                                button_label,
                                key=button_key,
                                on_click=handle_ai_suggested_action,
                                args=(action_code,),
                            )
                        else:
                            logger.warning(
                                f"Orphaned button label found: '{button_label}' in message: {message['content']}"
                            )
                            # If only label, display it as text to avoid losing it and prevent error.
                            st.markdown(f"[Button label: {button_label}]", unsafe_allow_html=True)

            if is_trainer and message["role"] == "assistant":
                with st.expander(_("✏️ Correct Response")):
                    message_id = f"msg_{message_idx}"  # Simpler ID
                    corrected_response = st.text_area(
                        _("Suggested correction:"),
                        value=message["content"],
                        key=f"correction_{message_id}",
                    )
                    if st.button(_("Save Correction"), key=f"save_{message_id}"):
                        # ... (existing correction saving logic) ...
                        correction_data = {
                            "timestamp": datetime.now().isoformat(),
                            "original": message["content"],
                            "corrected": corrected_response,
                        }

                        corrections_file = FEEDBACK_DIR / "corrections.csv"

                        user_prompt_content = ""
                        # Try to find the last user message for context
                        if (
                            message_idx > 0 and st.session_state.messages[message_idx - 1]["role"] == "user"
                        ):
                            user_prompt_content = st.session_state.messages[message_idx - 1]["content"]
                        elif len(st.session_state.messages) > 1 and st.session_state.messages[-2]["role"] == "user":
                            user_prompt_content = st.session_state.messages[-2]["content"]

                        if not corrections_file.exists():
                            with open(corrections_file, "w", newline="", encoding="utf-8") as f:
                                writer = csv.writer(f)
                                writer.writerow(
                                    [
                                        "timestamp",
                                        "user_message",
                                        "original_ai_response",
                                        "corrected_ai_response",
                                    ]
                                )

                        with open(corrections_file, "a", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            writer.writerow(
                                [
                                    correction_data["timestamp"],
                                    user_prompt_content,
                                    correction_data["original"],
                                    correction_data["corrected"],
                                ]
                            )
                        st.success(_("Correction saved successfully!"))

    # Persistent container for therapist booking button
    button_container = st.empty()
    if st.session_state.show_therapist_button:
        if button_container.button(_("Yes, help me book a therapist"), key="book_therapist_chat_area"):
            st.session_state.active_tab = "Therapist"
            st.rerun()

    # Chat input
    if prompt := st.chat_input(_("Type your message here...")):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()  # Rerun to display user message immediately

    # If the last message is from user, generate AI response
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                user_msg = st.session_state.messages[-1]["content"]
                history = [m for m in st.session_state.messages[:-1] if m["role"] in ("system", "assistant", "user")]

                # Crisis detection: generate immediate safety response locally
                if detect_critical_situation(user_msg):
                    reply = (
                        "I'm really sorry you're feeling like this. Your safety is the most important thing right now. "
                        "If you feel you might act on these thoughts, please reach out for immediate help:\n\n"
                        "• In the UAE: call 999 (police) or 998 (ambulance) for emergencies.\n"
                        "• Call 800 HOPE (800 4673) 24/7 to talk to a trained listener.\n"
                        "• If you're outside the UAE, please call your local emergency number or a trusted crisis line right away.\n\n"
                        "If you can, consider talking with someone you trust nearby. You don't have to face this alone. "
                        "I can also help you book a therapist session now if that feels helpful. "
                        "[BUTTON: Book a Therapist Session: action_navigate_therapist_booking]"
                    )
                    # Make the therapist-booking suggestion button visible
                    st.session_state.show_therapist_button = True
                else:
                    reply = CHAT_ENGINE.chat(st.session_state.session_id, history, user_msg)
                    full_response = reply
                message_placeholder.markdown(full_response, unsafe_allow_html=True)

                st.session_state.messages.append({"role": "assistant", "content": full_response})

                # After AI response, check if it made an APP_REQUEST
                app_request_match = re.search(app_request_regex, full_response)
                if app_request_match:
                    command = app_request_match.group(1)
                    params_str = app_request_match.group(2)
                    logger.info(f"AI made an APP_REQUEST: Command='{command}', Params_str='{params_str}'")
                    application_response_content = handle_ai_file_operation(command, params_str)
                    st.session_state.messages.append(
                        {
                            "role": "tool",
                            "name": "file_system_tool",
                            "content": application_response_content,
                        }
                    )
                    st.rerun()
                else:
                    st.rerun()

            except Exception as e:
                logger.error(f"Error with OpenAI API or processing AI response: {e}")
                st.error(_("Sorry, I encountered an error. Please try again."))
                # Avoid immediate rerun to prevent flicker

def render_exercise_tab():
    """Render the exercises tab with interactive guided exercises."""
    st.header(_("🧘 Guided Exercises"))
    
    # CSS for exercise buttons grid is now in style.css

    # Exercises are now defined globally in EXERCISES_DATA
    # No longer need: global EXERCISES_DATA
    # No longer need: if EXERCISES_DATA is None: ...

    active_exercise = st.session_state.get('active_exercise', None)

    if not active_exercise:
        st.subheader(_("Choose an Exercise"))

        # Create three columns for the exercises
        cols = st.columns(3)
        col_idx = 0

        for name, details in EXERCISES_DATA.items():
            with cols[col_idx % 3]:
                # Unique key for each button that simulates the card click
                button_key = f"exercise_card_btn_{name.replace(' ', '_')}"
                
                # Create HTML for the card
                card_html = f"""
                <div class="exercise-card" onclick="document.getElementById('button-{button_key}').click();">
                    <div class="exercise-icon">{details['icon']}</div>
                    <div class="exercise-name">{name}</div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
                
                # Hidden button that actually triggers the action
                with st.container(): # Use st.container to apply custom class if needed, or just rely on CSS for button itself
                    st.markdown('<div class="hidden-button-container">', unsafe_allow_html=True)
                    if st.button(name, key=button_key, help=f"Start {name}"):
                        set_active_exercise(name)
                        st.rerun() # Rerun to update the UI immediately
                    st.markdown('</div>', unsafe_allow_html=True)

            col_idx += 1
        
        # Add empty divs to fill grid if exercises are not a multiple of 3
        while col_idx % 3 != 0:
            with cols[col_idx % 3]:
                st.markdown("<div style='height: 150px;'></div>", unsafe_allow_html=True) # Placeholder
            col_idx += 1

    else:
        # Display the selected exercise
        if active_exercise in EXERCISES_DATA:
            if st.button(f"← {_('Back to Exercises')}"):
                set_active_exercise(None)
                st.rerun()
            else:
                st.subheader(active_exercise)
                st.markdown(f"<p>{EXERCISES_DATA[active_exercise]['desc']}</p>", unsafe_allow_html=True)
                if 'video_url' in EXERCISES_DATA[active_exercise]:
                    st.video(EXERCISES_DATA[active_exercise]['video_url'])
                EXERCISES_DATA[active_exercise]["func"]() # Call the rendering function for the exercise
        else:
            st.error(_("Selected exercise not found. Please choose another."))
            set_active_exercise(None) # Reset if something went wrong
            st.rerun()

    # Expander for exercise details - keep this if it's useful
    # with st.expander(_("Exercise Details"), expanded=False):
    # st.markdown(_("Additional information and instructions for the exercises can be found here."))
    # for name, details in EXERCISES_DATA.items():
    # st.markdown(f"**{name}**: {details['desc']}")

    st.markdown("---") # Existing separator

    # Button to discuss exercise content
    discuss_button_text = _("Discuss Exercise Content with AI")
    base_prompt_message = _("I'd like to discuss an exercise.")
    user_exercise_content = ""

    if active_exercise and active_exercise in EXERCISES_DATA:
        exercise_details = EXERCISES_DATA[active_exercise]
        user_exercise_content = f"Exercise: {active_exercise}. Description: {exercise_details['desc']}"

        if active_exercise == _("5-4-3-2-1 Grounding") and "grounding_items" in st.session_state:
            grounding_data = "\n".join([f"- {sense.capitalize()}: {', '.join(items) if items else 'None yet'}" for sense, items in st.session_state.grounding_items.items()])
            if grounding_data.strip(): # Check if there's actual grounding data
                 user_exercise_content += f"\nMy current grounding items:\n{grounding_data}"
        
        elif active_exercise == _("Likert Stress Check-in"):
            # Attempt to retrieve the last logged stress check-in for discussion
            # This assumes handle_exercise_logging stores it somewhere accessible or we can reconstruct it
            # For now, we'll just indicate the user is on this exercise.
            # A more robust solution would be to fetch the last entry from a log/DB.
            # For demonstration, let's check if form data is in session_state (if form isn't cleared post-submit)
            if "stress_checkin_data_for_discussion" in st.session_state: # Assume this is populated on form submit
                data = st.session_state.stress_checkin_data_for_discussion
                user_exercise_content += f"\nMy last stress check-in details:\nLevel: {data.get('level', 'N/A')}\nSymptoms: {', '.join(data.get('symptoms', []))}\nTriggers: {', '.join(data.get('triggers', []))}\nCoping: {', '.join(data.get('coping', []))}\nNotes: {data.get('notes', 'N/A')}"
            else:
                user_exercise_content += "\nI am currently on the Stress Check-in. I might have some data to discuss."


        elif active_exercise == _("Graded Exposure Planner") and "exposure_task" in st.session_state and "exposure_steps" in st.session_state:
            task = st.session_state.exposure_task
            steps = "\n".join([f"- {step['description']} (Anxiety: {step['anxiety_level']}/10)" for step in st.session_state.exposure_steps])
            if task or steps:
                user_exercise_content += f"\nMy exposure plan:\nTask: {task if task else 'Not defined'}\nSteps:\n{steps if steps else 'No steps defined'}"
        
        # If there is specific content, make a more pointed prompt
        if user_exercise_content != f"Exercise: {active_exercise}. Description: {exercise_details['desc']}": # Check if more than just desc
             final_prompt_message = _("I'd like to discuss the following from my '{ex_name}' exercise:\n{content_details}\nCan you help me with this?").format(ex_name=active_exercise, content_details=user_exercise_content)
        else: # Default if only description is available
            final_prompt_message = _("I'd like to discuss the '{ex_name}' exercise. Description: {ex_desc}. What are your thoughts or suggestions?").format(ex_name=active_exercise, ex_desc=exercise_details['desc'])

    else: # No active exercise, or not in the list
        final_prompt_message = base_prompt_message

    if st.button(discuss_button_text, key="discuss_exercise_content"):
        st.session_state.active_tab = "AI Chat"
        if "messages" not in st.session_state:
            st.session_state.messages = []
        st.session_state.messages.append({"role": "user", "content": final_prompt_message})
        # No need to set st.session_state.chat_input, as render_chat_tab will process st.session_state.messages
        st.rerun()

# Helper function to set active exercise
def set_active_exercise(name):
    # If navigating away from Likert Stress Check-in, clear its specific discussion data
    previous_exercise = st.session_state.get('active_exercise', None)
    if name is None and previous_exercise == _("Likert Stress Check-in"):
        if "stress_checkin_data_for_discussion" in st.session_state:
            del st.session_state.stress_checkin_data_for_discussion
            logger.debug("Cleared stress_checkin_data_for_discussion.")
            
    st.session_state.active_exercise = name

def render_therapist_tab():
    st.header(_("👩‍⚕️ Book Therapist Session"))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(_("Available Slots"))
        date = st.date_input(_("Select Date"))
        time_slot = st.selectbox(_("Select Time"), ["09:00", "10:00", "11:00", "14:00", "15:00"])
        therapist = st.selectbox(_("Select Therapist"), ["Dr. Smith", "Dr. Johnson", "Dr. Williams"])
        
        if st.button(_("Book Session")):
            st.success(_("Session booked successfully!"))
    
    with col2:
        st.subheader(_("Upcoming Sessions"))
        st.info(_("No upcoming sessions"))

def render_profile_tab():
    # st.header(_("👤 Profile Settings")) # Already handled by onboarding title if shown
    render_onboarding() # Display the onboarding/profile form

def render_admin_tab():
    st.header(_("📊 Admin Panel"))
    
    # Create tabs within the admin panel for better organization
    admin_tabs = st.tabs([_("Dashboard"), _("Knowledge Sources"), _("User Statistics"), _("System Settings"), _("Session Explorer"), _("Corrections Queue")])
    
    with admin_tabs[0]:  # Dashboard
        st.subheader(_("📈 MindShield Platform Dashboard"))
        
        # Show key metrics and stats
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            st.metric(label=_("Total Users"), value="132", delta="5")
        with metrics_col2:
            st.metric(label=_("Active Sessions"), value="43", delta="2")
        with metrics_col3:
            st.metric(label=_("Weekly Engagement"), value="78%", delta="12%")
        
        # Usage over time (simulated data)
        st.subheader(_("Platform Usage"))
        usage_data = pd.DataFrame({
            'Date': pd.date_range(start='2025-05-01', periods=14),
            'Active Users': [45, 52, 48, 55, 62, 58, 65, 72, 68, 75, 70, 80, 85, 82],
            'Sessions': [120, 135, 125, 140, 155, 145, 160, 175, 165, 180, 170, 190, 200, 195],
            'Exercises': [85, 92, 88, 95, 102, 98, 105, 112, 108, 115, 110, 120, 125, 122]
        })
        usage_data = usage_data.set_index('Date')
        st.line_chart(usage_data)
        
        # Popular features chart
        st.subheader(_("Most Used Features"))
        features = pd.DataFrame({
            'Feature': ['AI Chat', 'Box Breathing', 'Eisenhower Matrix', 'Grounding', 'PMR', 'Therapist Booking'],
            'Usage Count': [350, 210, 180, 160, 140, 120]
        })
        st.bar_chart(features.set_index('Feature'))
        
        # Recent activity log
        st.subheader(_("Recent Activity"))
        recent_data = [
            {"Time": "Today 14:32", "User": "User123", "Activity": "Completed 5-4-3-2-1 Grounding exercise"},
            {"Time": "Today 13:15", "User": "User456", "Activity": "Logged 30-min therapy session"},
            {"Time": "Today 11:47", "User": "User789", "Activity": "Updated Eisenhower Matrix"},
            {"Time": "Today 10:21", "User": "User234", "Activity": "Completed Box Breathing exercise"},
            {"Time": "Today 09:05", "User": "User567", "Activity": "Scheduled therapist appointment"}
        ]
        st.dataframe(recent_data)
    
    with admin_tabs[1]:  # Knowledge Sources
        st.subheader(_("📚 Upload Knowledge Source"))
        
        uploaded = st.file_uploader(_("Upload .pdf, .md, .txt"), type=["pdf", "md", "txt"])
        if uploaded:
            # Check file type using file extension
            file_extension = Path(uploaded.name).suffix.lower()
            
            if file_extension == '.pdf':
                with st.spinner("Processing PDF..."):
                    handle_pdf_upload(uploaded)
            else:
                # Handle other file types as before
                save_path = DATA_DIR / "data_library" / "subjects" / uploaded.name
                with open(save_path, "wb") as f:
                    f.write(uploaded.getbuffer())
                st.success(f"{uploaded.name} saved.")
        
        if st.button(_("Reload Library")):
            st.cache_data.clear()
            st.success("Reloaded.")
        
        st.subheader(_("Current files"))
        files_path = DATA_DIR / "data_library" / "subjects"
        if files_path.exists():
            for f in sorted(files_path.glob("*")):
                if f.suffix != '.analysis.txt':  # Don't show analysis files
                    st.text(f.name)
                    if f.suffix == '.pdf':
                        analysis_path = f.with_suffix('.analysis.txt')
                        if analysis_path.exists():
                            with st.expander("Show Analysis"):
                                st.text(analysis_path.read_text())
    
    with admin_tabs[2]:  # User Statistics
        st.subheader(_("👥 User Statistics"))
        
        # User growth chart
        user_growth = pd.DataFrame({
            'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
            'New Users': [32, 45, 52, 68, 85],
            'Total Users': [32, 77, 129, 197, 282]
        })
        st.subheader(_("User Growth"))
        st.line_chart(user_growth.set_index('Month'))
        
        # User engagement distribution
        st.subheader(_("User Engagement Distribution"))
        engagement_data = pd.DataFrame({
            'Engagement Level': ['High', 'Medium', 'Low', 'Inactive'],
            'Percentage': [35, 42, 18, 5]
        })
        st.bar_chart(engagement_data.set_index('Engagement Level'))
        
        # Most active users
        st.subheader(_("Most Active Users"))
        active_users = [
            {"User ID": "U12345", "Name": "Alex Chen", "Sessions": 48, "Exercises": 32},
            {"User ID": "U23456", "Name": "Sam Taylor", "Sessions": 42, "Exercises": 38},
            {"User ID": "U34567", "Name": "Jordan Lee", "Sessions": 39, "Exercises": 25},
            {"User ID": "U45678", "Name": "Casey Kim", "Sessions": 35, "Exercises": 30},
            {"User ID": "U56789", "Name": "Riley Smith", "Sessions": 34, "Exercises": 28}
        ]
        st.dataframe(active_users)
        
        # User feedback analysis
        st.subheader(_("User Satisfaction"))
        satisfaction = pd.DataFrame({
            'Rating': ['5 stars', '4 stars', '3 stars', '2 stars', '1 star'],
            'Percentage': [52, 38, 7, 2, 1]
        })
        st.bar_chart(satisfaction.set_index('Rating'))
    
    with admin_tabs[3]:  # System Settings
        st.subheader(_("⚙️ System Settings"))
        
        col1, col2 = st.columns(2)
        with col1:
            st.checkbox(_("Enable user registration"), value=True)
            st.checkbox(_("Allow anonymous usage"), value=False)
            st.checkbox(_("Enable email notifications"), value=True)
            st.checkbox(_("Debug mode"), value=False)
        
        with col2:
            st.number_input(_("Session timeout (minutes)"), min_value=5, max_value=120, value=30)
            st.selectbox(_("Default language"), CONFIG["SUPPORTED_LANGUAGES"], 
                        format_func=lambda x: CONFIG["LANGUAGE_NAMES"][x],
                        index=CONFIG["SUPPORTED_LANGUAGES"].index(CONFIG["DEFAULT_LANGUAGE"]))
            st.text_input(_("Support email"), value="support@mindshield.example.com")
        
        # API keys and integrations
        st.subheader(_("API Integrations"))
        openai_key = st.text_input(_("OpenAI API Key"), value="sk-*****", type="password")
        calendar_key = st.text_input(_("Calendar API Key"), value="cal-*****", type="password")
        
        if st.button(_("Save Settings"), use_container_width=True):
            st.success(_("Settings saved successfully!"))
            # In a real application, we would save these settings to a database

    # ---------------- Session Explorer -------------------
    with admin_tabs[4]:
        st.subheader(_("🗂️ Recent Chat Sessions"))
        try:
            from mindshield_core.logger import fetch_recent_logs
            logs = fetch_recent_logs(200)
            if logs:
                df = pd.DataFrame(logs)
                st.dataframe(df)
            else:
                st.info(_("No logs found yet."))
        except Exception as e:
            st.error(_("Could not load logs: {err}").format(err=str(e)))

    # ---------------- Corrections Queue ------------------
    with admin_tabs[5]:
        st.subheader(_("✏️ Corrections Queue"))
        corrections_file = FEEDBACK_DIR / "corrections.csv"
        if corrections_file.exists():
            df = pd.read_csv(corrections_file)
            st.dataframe(df)
        else:
            st.info(_("No corrections submitted yet."))

# ---------------- Add PHQ9_ITEMS and GAD7_ITEMS constants ----------------
PHQ9_ITEMS = [
    "Little interest or pleasure in doing things.",
    "Feeling down, depressed, or hopeless.",
    "Trouble falling or staying asleep, or sleeping too much.",
    "Feeling tired or having little energy.",
    "Poor appetite or overeating.",
    "Feeling bad about yourself — or that you're a failure.",
    "Trouble concentrating on things, e.g. reading or TV.",
    "Moving / speaking slowly or being fidgety / restless.",
    "Thoughts that you would be better off dead or hurting yourself."
]

GAD7_ITEMS = [
    "Feeling nervous, anxious or on edge.",
    "Not being able to stop or control worrying.",
    "Worrying too much about different things.",
    "Trouble relaxing.",
    "Being so restless it is hard to sit still.",
    "Becoming easily annoyed or irritable.",
    "Feeling afraid something awful might happen."
]
# -------------------------------------------------------------------------

def render_onboarding():
    """First-time (or profile) form – MindShield version."""
    st.header(_("Welcome to MindShield! 🌟"))
    st.write(_("Answer a few questions so we can match the right tools or therapist."))

    # MVP Skip option ------------------------------------------------------
    if st.button(_("Skip for now"), key="skip_onboarding"):
        st.session_state.profile_data["onboarding_completed"] = True
        st.session_state.profile_data["onboarding_date"] = datetime.now().isoformat()
        st.session_state.active_tab = "AI Chat"
        st.rerun()

    with st.form("onboarding_form"):
        # --- Section 0 · Consent & Language -------------------------------------------
        st.subheader(_("🔏 Consent & Language"))
        lang_col, consent_col = st.columns([1, 2])
        with lang_col:
            preferred_language = st.selectbox(
                _("App language"),
                CONFIG["SUPPORTED_LANGUAGES"],
                format_func=lambda x: CONFIG["LANGUAGE_NAMES"][x],
                index=CONFIG["SUPPORTED_LANGUAGES"].index(
                    st.session_state.profile_data.get(
                        "preferred_language", st.session_state.ui_lang
                    )
                ),
            )
        with consent_col:
            adult_ok   = st.checkbox(_("I confirm I am 18 years or older"), value=False)
            privacy_ok = st.checkbox(
                _("I understand MindShield is **not** an emergency service "
                  "and that data is stored locally for service improvement."),
                value=False,
            )

        # --- Section 1 · What brings you here? ----------------------------------------
        st.subheader(_("🎯 Main reason for joining"))
        main_reason = st.multiselect(
            _("Select up to three:"),
            [
                "Stress at work/studies", "General anxiety", "Panic attacks",
                "Trouble sleeping", "Low mood", "Relationship tension", "Grief",
                "Trauma flashbacks", "Building healthy habits", "Just exploring", "Other"
            ],
            max_selections=3,
            default=st.session_state.profile_data.get("main_reason", []),
        )
        other_reason = ""
        if "Other" in main_reason:
            other_reason = st.text_input(_("Tell us more (optional)"))

        best_outcome = st.text_input(
            _("If everything went well, how would MindShield help you?"),
            value=st.session_state.profile_data.get("best_outcome", ""),
        )

        # --- Section 2 · Basic profile -----------------------------------------------
        st.subheader(_("👤 About you"))
        name = st.text_input(_("First name or nickname"),
                             value=st.session_state.profile_data.get("name", ""))
        col_a, col_b = st.columns(2)
        with col_a:
            country = st.text_input(  # keep simple; you can swap to pycountry list later
                _("Country / Emirate"), value=st.session_state.profile_data.get("country", "")
            )
        with col_b:
            dob = st.date_input(
                _("Date of birth"),
                value=pd.to_datetime(
                    st.session_state.profile_data.get("dob", "1990-01-01")
                ),
            )

        gender = st.selectbox(
            _("Gender identity"),
            ["Woman", "Man", "Non-binary", _("Prefer not to say")],
            index=["Woman", "Man", "Non-binary", _("Prefer not to say")].index(
                st.session_state.profile_data.get("gender", _("Prefer not to say"))
            ),
        )
        relationship = st.selectbox(
            _("Relationship status"),
            ["Single", "In a relationship", "Married",
             "Divorced/Widowed", _("Prefer not to say")],
            index=0,
        )
        occupation = st.selectbox(
            _("Occupation"),
            ["Student", "Employed", "Self-employed",
             "Job-seeker", "Homemaker", "Retired"],
            index=0,
        )

        # --- Section 3 · Language & Culture ------------------------------------------
        st.subheader(_("🕌 Language & cultural match"))
        therapy_langs = st.multiselect(
            _("Preferred therapy language(s)"),
            ["English", "العربية", "Hindi", "Urdu", "Other"],
            default=st.session_state.profile_data.get("therapy_langs", []),
        )
        religion = st.selectbox(
            _("Religious / cultural context you'd like respected"),
            ["Islam", "Christianity", "Hinduism", "Buddhism",
             "Secular / No preference", _("Prefer not to say")],
            index=0,
        )

        # --- Section 4 · Past care & safety ------------------------------------------
        st.subheader(_("🛟 Past care & current safety"))
        had_therapy = st.radio(
            _("Have you ever spoken to a mental-health professional before?"), ["No", "Yes"]
        )
        on_meds = st.radio(
            _("Are you currently taking mental-health medication?"), ["No", "Yes"]
        )
        last_sh = st.selectbox(
            _("Last time you had thoughts of harming yourself"),
            ["Never", "> 12 mo", "3-12 mo", "1-3 mo", "2-4 wk", _("Past 2 wk")],
        )
        chronic_pain = st.text_input(
            _("Any chronic medical condition or pain? (optional)")
        )

        # --- Section 5 · Quick symptom check (PHQ-9 + GAD-7) -------------------------
        st.subheader(_("📋 Symptom check – last 2 weeks"))
        phq_scores = []
        for q in PHQ9_ITEMS:
            phq_scores.append(
                st.selectbox(
                    q, ["0 – Not at all", "1 – Several days",
                        "2 – > Half the days", "3 – Nearly every day"],
                    key=f"phq_{q}",
                )
            )
        gad_scores = []
        for q in GAD7_ITEMS:
            gad_scores.append(
                st.selectbox(
                    q, ["0 – Not at all", "1 – Several days",
                        "2 – > Half the days", "3 – Nearly every day"],
                    key=f"gad_{q}",
                )
            )
        daily_diff = st.selectbox(
            _("Overall, how difficult have these problems made day-to-day life?"),
            ["0 – Not difficult", "1 – Somewhat", "2 – Very", "3 – Extremely"],
        )

        # --- Section 6 · Lifestyle snapshot -----------------------------------------
        st.subheader(_("🏃 Lifestyle & wellbeing"))
        col_l1, col_l2, col_l3 = st.columns(3)
        with col_l1:
            phys_health = st.selectbox(_("Physical health"), ["Good", "Fair", "Poor"])
            sleep = st.selectbox(_("Sleep quality"), ["Good", "Fair", "Poor"])
        with col_l2:
            eating = st.selectbox(_("Eating habits"), ["Good", "Fair", "Poor"])
            alcohol = st.selectbox(
                _("Alcohol use"), ["Never", "Monthly or less", "Weekly", "Daily"]
            )
        with col_l3:
            nicotine = st.selectbox(
                _("Nicotine / vaping"), ["Never", "Occasionally", "Daily"]
            )
            exercise = st.selectbox(
                _("Exercise per week"), ["None", "1-2 x", "3-4 x", "5+ x"]
            )

        # --- Section 7 · Preferences -------------------------------------------------
        st.subheader(_("🎛️ Helper & tool preferences"))
        support_mode = st.radio(
            _("Preferred support mode (for now)"),
            ["Self-help AI only", "AI + live therapist", _("Not sure yet")],
        )
        comms = st.selectbox(
            _("If live sessions, how do you prefer to communicate?"),
            ["Messaging", "Audio", "Video", _("No preference")],
        )
        helper_style = st.slider(
            _("Helper style – Gentle ⟷ Direct"), 1, 5, 3
        )
        tool_interest = st.multiselect(
            _("Which tools interest you right now?"),
            ["Guided breathing", "Mood / habit tracking", "Psy-ed videos",
             "Therapist booking", "Peer support circles", "Journaling prompts", "None"],
        )

        # --- Section 8 · Marketing (optional) ---------------------------------------
        st.subheader(_("📣 How did you hear about us? (optional)"))
        source = st.selectbox(
            _("Source"), ["Friend / family", "Instagram", "TikTok",
                          "Google", "Radio", "Podcast", "Other"],
        )
        special_flags = st.multiselect(
            _("Tick any that apply (discounts)"),
            ["University student", "First-responder / military",
             "Person of determination", "Low income"],
        )

        # --- SUBMIT -----------------------------------------------------------------
        submitted = st.form_submit_button(
            _("Start My Journey") if not st.session_state.profile_data.get(
                "onboarding_completed") else _("Update profile")
        )

        if submitted:
            if not (adult_ok and privacy_ok):
                st.error(_("Please confirm age & privacy to continue."))
                st.stop()

            # Save everything ---------------------------------------------------------
            pdict = st.session_state.profile_data
            pdict.update({
                "preferred_language": preferred_language,
                "name": name, "country": country, "dob": str(dob),
                "gender": gender, "relationship": relationship, "occupation": occupation,
                "main_reason": [r for r in main_reason if r != "Other"] + (
                    [other_reason] if other_reason else []
                ),
                "best_outcome": best_outcome,
                "therapy_langs": therapy_langs, "religion": religion,
                "had_therapy": had_therapy, "on_meds": on_meds,
                "last_self_harm": last_sh, "chronic_pain": chronic_pain,
                "phq9_raw": phq_scores, "gad7_raw": gad_scores,
                "daily_diff": daily_diff,
                "phys_health": phys_health, "sleep": sleep, "eating": eating,
                "alcohol": alcohol, "nicotine": nicotine, "exercise": exercise,
                "support_mode": support_mode, "comms": comms,
                "helper_style": helper_style, "tool_interest": tool_interest,
                "source": source, "special_flags": special_flags,
                "onboarding_completed": True,
                "onboarding_date": datetime.now().isoformat(),
            })

            # switch UI language immediately
            st.session_state.ui_lang = preferred_language
            st.session_state.gettext_translations = LANGUAGES.get(
                preferred_language, LANGUAGES["en"]
            )

            st.success(_("Profile saved! Welcome aboard."))
            st.session_state.active_tab = "AI Chat"
            st.rerun()

def handle_exercise_logging(exercise: str, score: int, notes: str, mood_change: int = 0) -> None:
    """Handle exercise logging with error handling."""
    try:
        log_file = DATA_DIR / "logs" / "exercise_log.csv"
        with open(log_file, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                datetime.now().isoformat(),
                exercise,
                score,
                mood_change,
                notes
            ])
        
        st.success("Exercise logged successfully! 🎯")
        
        # Show progress over time
        if os.path.exists(log_file):
            df = pd.read_csv(log_file, names=["Date", "Exercise", "Score", "Mood Change", "Notes"])
            if len(df) > 1:
                # Exercise effectiveness over time
                st.subheader("Exercise Effectiveness")
                st.line_chart(df.set_index("Date")["Score"])
                st.caption("Your reported effectiveness scores over time")
                
                # Mood impact over time
                if not df["Mood Change"].isna().all():
                    st.subheader("Mood Impact")
                    st.line_chart(df.set_index("Date")["Mood Change"])
                    st.caption("How exercises affected your mood (-5 to +5)")
                
                # Exercise frequency
                st.subheader("Exercise Frequency")
                exercise_counts = df["Exercise"].value_counts()
                st.bar_chart(exercise_counts)
                st.caption("Number of times you've done each exercise")
    
    except Exception as e:
        logger.error(f"Error logging exercise: {e}")
        st.error("Failed to log exercise. Please try again.")

def get_profile_context():
    """Generate a short summary of the logged-in user's profile for AI context."""
    pdict = st.session_state.get("profile_data", {})
    if not pdict.get("onboarding_completed"):
        return ""  # Nothing yet

    # Build key details; keep it concise to save tokens
    summary_parts = []
    nickname = pdict.get("name")
    if nickname:
        summary_parts.append(f"Nickname: {nickname}")

    country = pdict.get("country")
    if country:
        summary_parts.append(f"Location: {country}")

    main_reason = pdict.get("main_reason")
    if main_reason:
        summary_parts.append("Reasons for joining: " + ", ".join(main_reason))

    best_outcome = pdict.get("best_outcome")
    if best_outcome:
        summary_parts.append(f"Goal: {best_outcome}")

    support_mode = pdict.get("support_mode")
    if support_mode:
        summary_parts.append(f"Preferred support mode: {support_mode}")

    helper_style = pdict.get("helper_style")
    if helper_style is not None:
        summary_parts.append(f"Helper style (1-gentle → 5-direct): {helper_style}")

    therapy_langs = pdict.get("therapy_langs")
    if therapy_langs:
        summary_parts.append("Therapy language(s): " + ", ".join(therapy_langs))

    # Any quick-screen scores – include total only to save space
    if pdict.get("phq9_raw"):
        try:
            phq_total = sum(int(x[0]) for x in pdict["phq9_raw"])  # each choice starts with digit 0-3
            summary_parts.append(f"PHQ-9 total: {phq_total}")
        except Exception:
            pass
    if pdict.get("gad7_raw"):
        try:
            gad_total = sum(int(x[0]) for x in pdict["gad7_raw"])
            summary_parts.append(f"GAD-7 total: {gad_total}")
        except Exception:
            pass

    return "USER PROFILE → " + " | ".join(summary_parts)

def main():
    """Main application function"""
    load_css("style.css") # Load CSS
    initialize_session_state() # Call initialization function
    st.title(_("🛡️ MindShield"))
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🌐 UI Language")
        sel = st.selectbox(
            "Select Language",
            CONFIG["SUPPORTED_LANGUAGES"],
            format_func=lambda x: CONFIG["LANGUAGE_NAMES"][x],
            index=CONFIG["SUPPORTED_LANGUAGES"].index(st.session_state.ui_lang),
            key="language_selector"
        )
        if sel != st.session_state.ui_lang:
            st.session_state.ui_lang = sel
            st.session_state.gettext_translations = LANGUAGES.get(sel, LANGUAGES["en"]) # Update translations
            st.rerun() # Rerun to apply changes

        # Navigation
        st.markdown("---")
        if st.session_state.profile_data.get("onboarding_completed", False):
            tabs = [_("AI Chat"), _("Exercises"), _("Therapist Booking"), _("Profile"), _("Admin Panel")]
            internal_tabs = ["AI Chat", "Exercises", "Therapist", "Profile", "Admin Panel"]
        else:
            tabs = [_("Home")]
            internal_tabs = ["Home"]
            
        label_to_internal = dict(zip(tabs, internal_tabs))
        
        # Ensure active_tab is valid
        if st.session_state.active_tab not in internal_tabs:
            st.session_state.active_tab = internal_tabs[0]
            
        tab_choice_label = st.radio(
            _("🧭 Navigate"), 
            tabs, 
            index=internal_tabs.index(st.session_state.active_tab),
            key="navigation_tabs"
        )
        st.session_state.active_tab = label_to_internal[tab_choice_label]

    # Main content area
    if st.session_state.active_tab == "Home" or not st.session_state.profile_data.get("onboarding_completed", False):
        render_onboarding()
    elif st.session_state.active_tab == "AI Chat":
        render_chat_tab()
    elif st.session_state.active_tab == "Exercises":
        render_exercise_tab()
    elif st.session_state.active_tab == "Therapist":
        render_therapist_tab()
    elif st.session_state.active_tab == "Profile":
        render_profile_tab()
    elif st.session_state.active_tab == "Admin Panel":
        render_admin_tab()

if __name__ == "__main__":
    main() 
