import streamlit as st
import pandas as pd
import joblib
import html as html_mod
st.set_page_config(
    page_title="Career Prediction AI",
    page_icon="CP",
    layout="wide",
    initial_sidebar_state="collapsed"
)
def _svg(path, size=18):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{path}</svg>'

ICON_BRAIN = _svg('<path d="M9.5 2A5.5 5.5 0 0 0 5 5.5c0 .3 0 .6.1.9A4.5 4.5 0 0 0 2 10.5 4.5 4.5 0 0 0 5 14.6V16a2 2 0 0 0 2 2h2"/><path d="M14.5 2A5.5 5.5 0 0 1 19 5.5c0 .3 0 .6-.1.9A4.5 4.5 0 0 1 22 10.5a4.5 4.5 0 0 1-3 4.1V16a2 2 0 0 1-2 2h-2"/><path d="M12 2v20"/>')
ICON_CODE = _svg('<polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/>')
ICON_TROPHY = _svg('<path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"/><path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"/><path d="M4 22h16"/><path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22"/><path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22"/><path d="M18 2H6v7a6 6 0 0 0 12 0V2Z"/>')
ICON_BOOK = _svg('<path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"/>')
ICON_AWARD = _svg('<circle cx="12" cy="8" r="6"/><path d="M15.477 12.89 17 22l-5-3-5 3 1.523-9.11"/>')
ICON_WRENCH = _svg('<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>')
ICON_MIC = _svg('<path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/>')
ICON_LIGHTBULB = _svg('<path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/>')
ICON_PEN = _svg('<path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/>')
ICON_CPU = _svg('<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/>')
ICON_BARCHART = _svg('<line x1="12" x2="12" y1="20" y2="10"/><line x1="18" x2="18" y1="20" y2="4"/><line x1="6" x2="6" y1="20" y2="16"/>')
ICON_ZAP = _svg('<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>')
ICON_USERS = _svg('<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>')
ICON_USER = _svg('<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>')
ICON_BOOKMARK = _svg('<path d="m19 21-7-4-7 4V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v16z"/>')
ICON_COMPASS = _svg('<circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/>')
ICON_BUILDING = _svg('<rect width="16" height="20" x="4" y="2" rx="2" ry="2"/><path d="M9 22v-4h6v4"/><path d="M8 6h.01"/><path d="M16 6h.01"/><path d="M12 6h.01"/><path d="M12 10h.01"/><path d="M12 14h.01"/><path d="M16 10h.01"/><path d="M16 14h.01"/><path d="M8 10h.01"/><path d="M8 14h.01"/>')
ICON_GRADUATION = _svg('<path d="M22 10v6M2 10l10-5 10 5-10 5z"/><path d="M6 12v5c0 2 6 3 6 3s6-1 6-3v-5"/>')
ICON_BOOKOPEN = _svg('<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>')
ICON_TARGET = _svg('<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>')
ICON_SHIELD = _svg('<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>')
ICON_SEARCH = _svg('<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>')
ICON_PALETTE = _svg('<circle cx="13.5" cy="6.5" r=".5"/><circle cx="17.5" cy="10.5" r=".5"/><circle cx="8.5" cy="7.5" r=".5"/><circle cx="6.5" cy="11.5" r=".5"/><path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10c.926 0 1.648-.746 1.648-1.688 0-.437-.18-.835-.437-1.125-.29-.289-.438-.652-.438-1.125a1.64 1.64 0 0 1 1.668-1.668h1.996c3.051 0 5.555-2.503 5.555-5.554C21.965 6.012 17.461 2 12 2z"/>')
ICON_ROCKET = _svg('<path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z"/><path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z"/><path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0"/><path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5"/>')
ICON_LAPTOP = _svg('<path d="M20 16V7a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v9m16 0H4m16 0 1.28 2.55a1 1 0 0 1-.9 1.45H3.62a1 1 0 0 1-.9-1.45L4 16"/>')

FEATURE_META = {
    'Logical quotient rating': {
        'label': 'Logical Thinking',
        'desc': 'Rate your logical reasoning ability (1–10)',
        'icon': ICON_BRAIN
    },
    'hackathons': {
        'label': 'Hackathon Experience',
        'desc': 'Number of hackathons you\'ve participated in',
        'icon': ICON_TROPHY
    },
    'coding skills rating': {
        'label': 'Coding Skills',
        'desc': 'How strong are your programming skills? (1–10)',
        'icon': ICON_CODE
    },
    'Extra-courses did': {
        'label': 'Extra Courses',
        'desc': 'Have you taken additional courses outside your curriculum?',
        'icon': ICON_BOOK
    },
    'certifications': {
        'label': 'Certifications',
        'desc': 'Area of your professional certifications',
        'icon': ICON_AWARD
    },
    'workshops': {
        'label': 'Workshop Attendance',
        'desc': 'Type of workshops you\'ve attended',
        'icon': ICON_WRENCH
    },
    'public speaking points': {
        'label': 'Public Speaking',
        'desc': 'Rate your public speaking confidence (1–10)',
        'icon': ICON_MIC
    },
    'self-learning capability?': {
        'label': 'Self-Learning',
        'desc': 'Are you a self-directed learner?',
        'icon': ICON_LIGHTBULB
    },
    'reading and writing skills': {
        'label': 'Reading & Writing',
        'desc': 'How would you rate your literacy skills?',
        'icon': ICON_PEN
    },
    'memory capability score': {
        'label': 'Memory Capability',
        'desc': 'How strong is your memory retention?',
        'icon': ICON_CPU
    },
    'Management or Technical': {
        'label': 'Career Orientation',
        'desc': 'Do you lean towards management or technical roles?',
        'icon': ICON_BARCHART
    },
    'hard/smart worker': {
        'label': 'Work Style',
        'desc': 'Do you identify more as a hard worker or smart worker?',
        'icon': ICON_ZAP
    },
    'worked in teams ever?': {
        'label': 'Team Experience',
        'desc': 'Have you worked in team-based environments?',
        'icon': ICON_USERS
    },
    'Introvert': {
        'label': 'Introvert',
        'desc': 'Would you describe yourself as an introvert?',
        'icon': ICON_USER
    },
    'Interested subjects': {
        'label': 'Subject Interest',
        'desc': 'Which subject area interests you the most?',
        'icon': ICON_BOOKMARK
    },
    'interested career area ': {
        'label': 'Career Interest',
        'desc': 'Which career area are you drawn towards?',
        'icon': ICON_COMPASS
    },
    'Type of company want to settle in?': {
        'label': 'Ideal Company Type',
        'desc': 'What kind of company do you envision working at?',
        'icon': ICON_BUILDING
    },
    'Taken inputs from seniors or elders': {
        'label': 'Mentorship',
        'desc': 'Have you sought guidance from seniors or mentors?',
        'icon': ICON_GRADUATION
    },
    'Interested Type of Books': {
        'label': 'Reading Preferences',
        'desc': 'What genre of books do you enjoy most?',
        'icon': ICON_BOOKOPEN
    },
}
CAREER_META = {
    "Software Development": {
        "icon": ICON_LAPTOP,
        "color": "#4f46e5",
        "desc": "Build the applications and systems that power the digital world — from web platforms to enterprise software.",
    },
    "Cybersecurity": {
        "icon": ICON_SHIELD,
        "color": "#dc2626",
        "desc": "Protect organizations from digital threats, manage security infrastructure, and respond to incidents.",
    },
    "QA & Support": {
        "icon": ICON_SEARCH,
        "color": "#059669",
        "desc": "Ensure software quality through testing, support users, and bridge the gap between development and end users.",
    },
    "Design": {
        "icon": ICON_PALETTE,
        "color": "#d946ef",
        "desc": "Craft beautiful, intuitive user experiences and interfaces that delight millions of users.",
    },
}
DEFAULT_CAREER_META = {
    "icon": ICON_ROCKET,
    "color": "#4f46e5",
    "desc": "An exciting career path with great growth potential. Keep exploring and building your skills!",
}
RECOMMENDATIONS = {
    "Software Development": [
        ("Master Data Structures & Algorithms", "Foundation of every coding interview and system design."),
        ("Learn a Web/Mobile Framework", "React, Flutter, or Django — pick one and go deep."),
        ("Build End-to-End Projects", "Ship real products to stand out from the crowd."),
        ("Learn System Design", "Understand how large-scale systems are architected."),
    ],
    "Cybersecurity": [
        ("Get CompTIA Security+ / CISSP", "Industry-standard certifications that open doors."),
        ("Learn Ethical Hacking", "Hands-on penetration testing with tools like Burp Suite."),
        ("Understand Networking Deeply", "TCP/IP, DNS, firewalls — the core of all security."),
        ("Study Incident Response", "Learn how to detect, contain, and recover from breaches."),
    ],
    "QA & Support": [
        ("Learn Test Automation", "Selenium, Cypress, or Playwright for efficient testing."),
        ("Master Communication", "Technical troubleshooting starts with clear communication."),
        ("Understand Business Processes", "QA is about validating business logic, not just code."),
        ("Learn Python/Bash Scripting", "Automate repetitive tasks and build test utilities."),
    ],
    "Design": [
        ("Master Figma / Adobe XD", "Industry-standard design tools for prototyping."),
        ("Study User Psychology", "Great UX starts with understanding human behavior."),
        ("Build a UI Portfolio", "Showcase your best work on Dribbble or Behance."),
        ("Learn Basic HTML/CSS", "Bridge the gap between design and development."),
    ],
}

DEFAULT_RECOMMENDATIONS = [
    ("Keep Learning & Exploring", "Stay curious about new technologies and trends."),
    ("Build a Strong Portfolio", "Showcase your projects and skills to stand out."),
    ("Seek Mentorship", "Connect with professionals in your target field."),
    ("Get Certified", "Industry certifications validate your expertise."),
]
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ══════════════════════════════════════════════
       CSS CUSTOM PROPERTIES — Clean White Theme
       ══════════════════════════════════════════════ */
    :root {
        --bg-primary: #ffffff;
        --bg-surface: #ffffff;
        --bg-surface-hover: #f9fafb;
        --bg-input: #f3f4f6;
        --bg-input-hover: #e5e7eb;
        --border-default: #e5e7eb;
        --border-subtle: #f3f4f6;
        --border-muted: #d1d5db;
        --text-primary: #111827;
        --text-body: #374151;
        --text-muted: #6b7280;
        --text-subtle: #9ca3af;
        --text-faint: #d1d5db;
        --accent: #4f46e5;
        --accent-hover: #4338ca;
        --accent-bg: rgba(79, 70, 229, 0.06);
        --accent-border: rgba(79, 70, 229, 0.15);
        --accent-focus: rgba(79, 70, 229, 0.12);
        --rec-bg: #f9fafb;
        --rec-border: #e5e7eb;
        --rec-hover-bg: #ffffff;
        --rec-hover-border: #d1d5db;
        --step-inactive-bg: #f3f4f6;
        --step-inactive-color: #9ca3af;
        --step-inactive-border: #e5e7eb;
    }

    /* ── Reset & Base ──────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: var(--bg-primary) !important;
        color: var(--text-body);
    }

    /* ── Hide Streamlit Defaults ───────────────── */
    .block-container { max-width: 1100px; padding-top: 1rem; }

    /* ── Typography ────────────────────────────── */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }

    /* ── Hero ───────────────────────────────────── */
    .hero {
        text-align: center;
        padding: 1.5rem 1rem 0.5rem;
        animation: fadeIn 0.5s ease both;
    }
    .hero-badge {
        display: inline-block;
        background: var(--accent-bg);
        border: 1px solid var(--accent-border);
        color: var(--accent);
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        padding: 0.3rem 0.9rem;
        border-radius: 100px;
        margin-bottom: 1rem;
    }
    .hero h1 {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.15;
        margin: 0 0 0.75rem;
    }
    .hero h1 span {
        color: var(--accent);
    }
    .hero p {
        color: var(--text-muted);
        font-size: 1rem;
        line-height: 1.6;
        max-width: 480px;
        margin: 0 auto;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ── Decorative Divider ─────────────────────── */
    .divider {
        width: 48px;
        height: 2px;
        background: var(--accent);
        border-radius: 10px;
        margin: 0.75rem auto 1rem;
    }

    /* ── Progress Bar ──────────────────────────── */
    .progress-container {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0;
        margin: 0 auto 1rem;
        max-width: 520px;
    }
    .step-node {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 0.82rem;
        flex-shrink: 0;
        transition: all 0.3s ease;
    }
    .step-node.active {
        background: var(--accent);
        color: white;
    }
    .step-node.completed {
        background: var(--accent);
        color: white;
    }
    .step-node.inactive {
        background: var(--step-inactive-bg);
        color: var(--step-inactive-color);
        border: 1px solid var(--step-inactive-border);
    }
    .step-connector {
        height: 2px;
        width: 72px;
        flex-shrink: 0;
        transition: background 0.3s ease;
    }
    .step-connector.done {
        background: var(--accent);
    }
    .step-connector.pending {
        background: var(--step-inactive-bg);
    }
    .step-label-row {
        display: flex;
        justify-content: center;
        gap: 42px;
        margin-top: 0.4rem;
        margin-bottom: 0.75rem;
        max-width: 520px;
        margin-left: auto;
        margin-right: auto;
    }
    .step-label {
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 600;
        text-align: center;
        width: 100px;
    }
    .step-label.active { color: var(--accent); }
    .step-label.inactive { color: var(--text-subtle); }
    .step-label.completed { color: var(--text-muted); }

    /* ── Section Header ────────────────────────── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1rem;
    }
    .section-icon {
        width: 40px;
        height: 40px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: var(--accent-bg);
        border: 1px solid var(--accent-border);
        color: var(--accent);
    }
    .section-icon svg {
        width: 20px;
        height: 20px;
    }
    .section-header h2 {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
        line-height: 1.2;
    }
    .section-header p {
        font-size: 0.8rem;
        color: var(--text-muted);
        margin: 0;
    }

    /* ── Input Card ─────────────────────────────── */
    .input-card {
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
        border-radius: 12px 12px 0 0;
        padding: 0.75rem 1rem 0.35rem;
        margin-bottom: -1rem;
        transition: all 0.2s ease;
    }
    .input-card:hover {
        border-color: var(--border-subtle);
    }
    .input-label {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.1rem;
    }
    .input-label .icon {
        display: flex;
        align-items: center;
        color: var(--text-muted);
    }
    .input-label .icon svg {
        width: 16px;
        height: 16px;
    }
    .input-label .text {
        font-weight: 600;
        font-size: 0.85rem;
        color: var(--text-primary);
    }
    .input-desc {
        font-size: 0.72rem;
        color: var(--text-subtle);
        margin-bottom: 0.4rem;
        padding-left: 1.5rem;
    }

    /* ── Streamlit Input Overrides ──────────────── */
    div[data-baseweb="select"] > div {
        background: var(--bg-input) !important;
        border: 1px solid var(--border-default) !important;
        border-radius: 8px !important;
        color: var(--text-body) !important;
        transition: all 0.2s ease !important;
    }
    div[data-baseweb="select"]:hover > div {
        border-color: var(--border-subtle) !important;
    }
    div[data-baseweb="select"]:focus-within > div {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px var(--accent-focus) !important;
    }

    input[type="number"] {
        background: var(--bg-input) !important;
        border: 1px solid var(--border-default) !important;
        border-radius: 8px !important;
        color: var(--text-body) !important;
        transition: all 0.2s ease !important;
    }
    input[type="number"]:hover {
        border-color: var(--border-subtle) !important;
    }
    input[type="number"]:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px var(--accent-focus) !important;
    }

    /* Hide default streamlit labels (we render our own) */
    .stSelectbox label, .stNumberInput label {
        display: none !important;
    }

    /* ── Navigation Buttons ────────────────────── */
    .nav-row {
        display: flex;
        gap: 1rem;
        justify-content: space-between;
        margin-top: 1.5rem;
    }

    /* Next / Predict button */
    .stButton > button[kind="primary"],
    div[data-testid="stButton"] > button {
        width: 100%;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0.75rem 2rem;
        border: none;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .predict-btn > button {
        background: var(--accent) !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        padding: 0.85rem 2rem !important;
        border: none !important;
        border-radius: 10px !important;
        width: 100% !important;
        margin-top: 0.5rem !important;
    }
    .predict-btn > button:hover {
        background: var(--accent-hover) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.25) !important;
    }
    .predict-btn > button:active {
        transform: translateY(0px) !important;
    }

    .nav-btn-secondary > button {
        background: var(--bg-input) !important;
        color: var(--text-muted) !important;
        border: 1px solid var(--border-default) !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        width: 100% !important;
    }
    .nav-btn-secondary > button:hover {
        background: var(--bg-input-hover) !important;
        color: var(--text-body) !important;
        border-color: var(--border-subtle) !important;
    }

    .nav-btn-primary > button {
        background: var(--accent) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 10px !important;
        width: 100% !important;
    }
    .nav-btn-primary > button:hover {
        background: var(--accent-hover) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2) !important;
    }

    /* ── Result Card ───────────────────────────── */
    .result-card {
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin-top: 2rem;
        animation: fadeIn 0.5s ease both;
        position: relative;
        overflow: hidden;
    }
    .result-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        border-radius: 16px 16px 0 0;
    }
    .result-icon {
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .result-icon svg {
        width: 48px;
        height: 48px;
    }
    .result-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: var(--text-subtle);
        margin-bottom: 0.5rem;
    }
    .result-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
        line-height: 1.2;
    }
    .result-desc {
        color: var(--text-muted);
        font-size: 0.92rem;
        line-height: 1.6;
        max-width: 460px;
        margin: 0 auto 2rem;
    }

    /* Recommendation Steps */
    .rec-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.65rem;
        max-width: 540px;
        margin: 0 auto;
        text-align: left;
    }
    .rec-step {
        background: var(--rec-bg);
        border: 1px solid var(--rec-border);
        border-radius: 10px;
        padding: 0.9rem;
        transition: all 0.2s ease;
    }
    .rec-step:hover {
        border-color: var(--rec-hover-border);
        background: var(--rec-hover-bg);
        transform: translateY(-1px);
    }
    .rec-num {
        font-weight: 700;
        font-size: 1.2rem;
        opacity: 0.12;
        margin-bottom: 0.2rem;
    }
    .rec-step-title {
        font-weight: 600;
        font-size: 0.82rem;
        color: var(--text-primary);
        margin-bottom: 0.2rem;
    }
    .rec-step-desc {
        font-size: 0.72rem;
        color: var(--text-muted);
        line-height: 1.45;
    }

    .next-steps-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: var(--text-subtle);
        margin-bottom: 1rem;
        text-align: center;
    }

    .start-over-btn > button {
        background: transparent !important;
        color: var(--text-muted) !important;
        border: 1px solid var(--border-default) !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        width: auto !important;
        margin: 2rem auto 0 !important;
        padding: 0.6rem 2rem !important;
    }
    .start-over-btn > button:hover {
        color: var(--text-body) !important;
        border-color: var(--border-subtle) !important;
        background: var(--bg-input) !important;
    }

    /* ── Footer ────────────────────────────────── */
    .app-footer {
        text-align: center;
        padding: 1.5rem 0 1rem;
        color: var(--text-faint);
        font-size: 0.72rem;
        letter-spacing: 0.3px;
    }

    /* ── Responsive ────────────────────────────── */
    @media (max-width: 640px) {
        .hero h1 { font-size: 1.8rem; }
        .rec-grid { grid-template-columns: 1fr; }
        .result-title { font-size: 1.5rem; }
        .block-container { padding-left: 1rem; padding-right: 1rem; }
    }
</style>
""", unsafe_allow_html=True)
class CareerModelWrapper:
    """Wrapper that handles preprocessing, prediction, and label decoding seamlessly."""
    def __init__(self, preprocessor, model, label_encoder):
        self.preprocessor = preprocessor
        self.model = model
        self.label_encoder = label_encoder

    def predict(self, X):
        X_proc = self.preprocessor.transform(X)
        preds = self.model.predict(X_proc)
        return self.label_encoder.inverse_transform(preds)


@st.cache_resource
def load_model_and_data():
    try:
        model = joblib.load("models/career_model_grouped.pkl")
    except FileNotFoundError:
        model = joblib.load("models/career_model.pkl")
    sample = pd.read_csv("data/PS2_Dataset.csv")
    features = sample.drop(columns=["Suggested Job Role"])
    return model, features


model, features = load_model_and_data()
STEPS = [
    {
        "key": "tech",
        "title": "Technical Skills",
        "subtitle": "Tell us about your technical abilities and learning.",
        "icon": ICON_CODE,
        "features": [
            'Logical quotient rating', 'hackathons',
            'coding skills rating', 'Extra-courses did',
            'certifications', 'workshops',
        ],
    },
    {
        "key": "soft",
        "title": "Personality & Soft Skills",
        "subtitle": "How you work, communicate, and think.",
        "icon": ICON_BRAIN,
        "features": [
            'public speaking points', 'self-learning capability?',
            'reading and writing skills', 'memory capability score',
            'Management or Technical', 'hard/smart worker',
            'worked in teams ever?', 'Introvert',
        ],
    },
    {
        "key": "interests",
        "title": "Interests & Goals",
        "subtitle": "What excites you and where you want to go.",
        "icon": ICON_TARGET,
        "features": [
            'Interested subjects', 'interested career area ',
            'Type of company want to settle in?',
            'Taken inputs from seniors or elders',
            'Interested Type of Books',
        ],
    },
]
if "current_step" not in st.session_state:
    st.session_state.current_step = 0
if "show_result" not in st.session_state:
    st.session_state.show_result = False
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "user_data" not in st.session_state:
    st.session_state.user_data = {}
st.markdown("""
<div class="hero">
    <div class="hero-badge">AI-Powered Career Guidance</div>
    <h1>Find Your <span>Perfect Career</span></h1>
    <p>Answer a few questions about your skills, personality, and interests — our ML model will predict your ideal tech career path.</p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)
current = st.session_state.current_step
step_labels = ["Technical", "Personality", "Interests"]

nodes_html = ""
for i in range(3):
    if i > 0:
        conn_class = "done" if i <= current else "pending"
        nodes_html += f'<div class="step-connector {conn_class}"></div>'
    if i < current:
        cls = "completed"
        inner = "✓"
    elif i == current:
        cls = "active"
        inner = str(i + 1)
    else:
        cls = "inactive"
        inner = str(i + 1)
    nodes_html += f'<div class="step-node {cls}">{inner}</div>'

labels_html = ""
for i, lbl in enumerate(step_labels):
    if i < current:
        labels_html += f'<div class="step-label completed">{lbl}</div>'
    elif i == current:
        labels_html += f'<div class="step-label active">{lbl}</div>'
    else:
        labels_html += f'<div class="step-label inactive">{lbl}</div>'

if not st.session_state.show_result:
    st.markdown(f"""
    <div class="progress-container">{nodes_html}</div>
    <div class="step-label-row">{labels_html}</div>
    """, unsafe_allow_html=True)
user_input = {}


def render_step_inputs(step_info):
    """Render inputs for a given step with custom card wrappers."""
    st.markdown(f"""
    <div class="section-header">
        <div class="section-icon">{step_info['icon']}</div>
        <div>
            <h2>{step_info['title']}</h2>
            <p>{step_info['subtitle']}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(2)
    for i, col_name in enumerate(step_info['features']):
        meta = FEATURE_META.get(col_name, {
            'label': col_name.title(),
            'desc': '',
            'icon': ICON_BOOKMARK
        })
        with cols[i % 2]:
            st.markdown(f"""
            <div class="input-card">
                <div class="input-label">
                    <span class="icon">{meta['icon']}</span>
                    <span class="text">{meta['label']}</span>
                </div>
                <div class="input-desc">{meta['desc']}</div>
            </div>
            """, unsafe_allow_html=True)

            if not pd.api.types.is_numeric_dtype(features[col_name]):
                opts = sorted(features[col_name].dropna().unique())
                default_val = st.session_state.user_data.get(col_name, opts[0])
                default_idx = opts.index(default_val) if default_val in opts else 0
                val = st.selectbox(
                    meta['label'],
                    opts,
                    index=default_idx,
                    key=f"input_{col_name}",
                )
            else:
                default_val = st.session_state.user_data.get(col_name, float(features[col_name].median()))
                val = st.number_input(
                    meta['label'],
                    value=float(default_val),
                    min_value=0.0,
                    step=1.0,
                    key=f"input_{col_name}",
                )
            st.session_state.user_data[col_name] = val
if not st.session_state.show_result:
    render_step_inputs(STEPS[current])
    nav_cols = st.columns([1, 1] if current > 0 else [1])

    if current > 0:
        with nav_cols[0]:
            st.markdown('<div class="nav-btn-secondary">', unsafe_allow_html=True)
            if st.button("← Back", key="btn_back"):
                st.session_state.current_step -= 1
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    if current < 2:
        with nav_cols[-1]:
            st.markdown('<div class="nav-btn-primary">', unsafe_allow_html=True)
            if st.button("Next →", key="btn_next"):
                st.session_state.current_step += 1
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        with nav_cols[-1]:
            st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
            if st.button("Predict My Career", key="btn_predict"):
                inp = pd.DataFrame([st.session_state.user_data])
                pred = model.predict(inp)[0]
                st.session_state.prediction = pred
                st.session_state.show_result = True
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

else:
    pred = st.session_state.prediction
    meta = CAREER_META.get(pred, DEFAULT_CAREER_META)
    recs = RECOMMENDATIONS.get(pred, DEFAULT_RECOMMENDATIONS)
    rec_html = ""
    for i, (title, desc) in enumerate(recs):
        safe_title = html_mod.escape(title)
        safe_desc = html_mod.escape(desc)
        rec_html += (
            f'<div class="rec-step">'
            f'<div class="rec-num" style="color: {meta["color"]};">0{i+1}</div>'
            f'<div class="rec-step-title">{safe_title}</div>'
            f'<div class="rec-step-desc">{safe_desc}</div>'
            f'</div>'
        )

    safe_pred = html_mod.escape(pred)
    safe_desc = html_mod.escape(meta['desc'])
    st.markdown(
        f'<div class="result-card" style="--accent: {meta["color"]};">'
        f'<div style="position:absolute;top:0;left:0;right:0;height:3px;'
        f'background:{meta["color"]};border-radius:16px 16px 0 0;"></div>'
        f'<div class="result-icon">{meta["icon"]}</div>'
        f'<div class="result-label">Your Ideal Career Path</div>'
        f'<div class="result-title" style="color: {meta["color"]};">{safe_pred}</div>'
        f'<div class="result-desc">{safe_desc}</div>'
        f'<div class="next-steps-label">Recommended Next Steps</div>'
        f'<div class="rec-grid">{rec_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")
    col_center = st.columns([1, 2, 1])[1]
    with col_center:
        st.markdown('<div class="start-over-btn">', unsafe_allow_html=True)
        if st.button("Start Over", key="btn_reset"):
            st.session_state.current_step = 0
            st.session_state.show_result = False
            st.session_state.prediction = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
st.markdown("""
<div class="app-footer">
    Built with Streamlit &middot; ML-Powered Career Prediction
</div>
""", unsafe_allow_html=True)
