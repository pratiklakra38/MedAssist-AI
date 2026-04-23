import streamlit as st
import json
import os
from src.inference import Pipeline
from src.agents import QuestionBot, ReviewerBot

st.set_page_config(page_title="MedAssist AI", page_icon="🩺", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0f0f0f; color: #e0e0e0; }
.title { text-align:center; font-size:2.4rem; font-weight:700;
    background: linear-gradient(135deg, #4fc3f7, #ab47bc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.subtitle { text-align:center; color:#888; font-size:0.95rem; margin-bottom:2rem; }
.card { background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
    border-radius:16px; padding:1.5rem; margin-bottom:1rem; }
.badge { display:inline-block; padding:4px 14px; border-radius:20px; font-weight:600; font-size:0.9rem; }
.high { background:rgba(244,67,54,0.2); color:#ef5350; }
.moderate { background:rgba(255,193,7,0.15); color:#ffc107; }
.low { background:rgba(76,175,80,0.15); color:#66bb6a; }
.warn { background:rgba(255,152,0,0.1); border:1px solid rgba(255,152,0,0.3);
    border-radius:12px; padding:1rem; color:#ffb74d; margin:1rem 0; }
.agent { background:rgba(171,71,188,0.08); border:1px solid rgba(171,71,188,0.25);
    border-radius:12px; padding:1rem; margin:0.5rem 0; }
.tag { display:inline-block; background:rgba(79,195,247,0.1); border:1px solid rgba(79,195,247,0.2);
    color:#4fc3f7; padding:3px 10px; border-radius:12px; font-size:0.8rem; margin:2px; }
.note { background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06);
    border-radius:12px; padding:1rem; text-align:center; color:#888; font-size:0.85rem; margin-top:2rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">MedAssist AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Hybrid AI Clinical Screening &bull; Rule-Based + ML + Agentic AI</div>', unsafe_allow_html=True)

@st.cache_resource
def load_all():
    return Pipeline(), QuestionBot(), ReviewerBot()

pipe, questioner, reviewer = load_all()

with open(os.path.join("data", "processed", "symptom_list.json")) as f:
    symptom_list = json.load(f)

pretty = [s.replace("_", " ").title() for s in symptom_list]

st.markdown("### Select Your Symptoms")
picked = st.multiselect("Choose all symptoms you are experiencing:", pretty, placeholder="Start typing...")
st.button("🔍 Analyze", use_container_width=True, type="primary", key="go")

vec = [0] * len(symptom_list)
for s in picked:
    vec[pretty.index(s)] = 1

if "pending_q" not in st.session_state:
    st.session_state.pending_q = None
if "answered" not in st.session_state:
    st.session_state.answered = False

if st.session_state.go and sum(vec) > 0:
    st.session_state.answered = False
    q = questioner.check(vec)
    st.session_state.pending_q = q

if st.session_state.go and sum(vec) == 0:
    st.warning("Pick at least one symptom.")

if st.session_state.pending_q and not st.session_state.answered:
    q = st.session_state.pending_q
    st.markdown(f'''
    <div class="agent">
        <strong>🤖 Virtual Doctor</strong><br>
        Your symptoms could match <b>{q["confused_between"][0]}</b> or
        <b>{q["confused_between"][1]}</b>.<br><br>
        <em>{q["question"]}</em>
    </div>''', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Yes", use_container_width=True):
            if q["symptom_index"] is not None:
                vec[q["symptom_index"]] = 1
            st.session_state.answered = True
            st.rerun()
    with c2:
        if st.button("No", use_container_width=True):
            st.session_state.answered = True
            st.rerun()

if (st.session_state.go or st.session_state.answered) and sum(vec) > 0:
    result = pipe.run(vec)
    result = reviewer.check(result)

    if "warning" in result:
        st.markdown(f'<div class="warn"><strong>🔍 Reviewer:</strong> {result["warning"]}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Results")

    for i, p in enumerate(result["predictions"]):
        urg = p["urgency"].lower()
        tags = ""
        if p["matched_symptoms"]:
            tags = "<div style='margin-top:8px'>Rule Matches: " + "".join(
                f'<span class="tag">{s.replace("_"," ").title()}</span>' for s in p["matched_symptoms"]
            ) + "</div>"

        st.markdown(f'''
        <div class="card">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <div><span style="color:#666">#{i+1}</span>
                    <span style="font-size:1.3rem;font-weight:600;color:#fff">&nbsp;{p["disease"]}</span></div>
                <div><span class="badge {urg}">{p["urgency"]}</span>
                    &nbsp;<span style="font-size:1.1rem;font-weight:600">{p["confidence"]}%</span></div>
            </div>{tags}
        </div>''', unsafe_allow_html=True)

    st.markdown(f'<div class="note">⚠️ {result["disclaimer"]}</div>', unsafe_allow_html=True)

    if st.session_state.answered:
        st.session_state.pending_q = None
