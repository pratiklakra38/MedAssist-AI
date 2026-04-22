"""
MedAssist AI — Streamlit Web Interface

Frontend for the Autonomous Medical Triage Agent.
Implements the User Interface → Symptom Collector → Agent → Diagnosis Output pipeline
described in the Final Report's system architecture.

Features:
- Modern dark glassmorphism UI with CSS animations
- Patient symptom collector (multiselect)
- Top 3 ranked diagnoses display
- Agent reasoning transparency panel
- Triage urgency badges with pulse animations
- Actionable next steps
- Mandatory ethical disclaimer
"""

import streamlit as st
import time
from src.inference import MedicalTriageAgent

@st.cache_resource
def load_agent():
    """Cache the agent so ML model weights are loaded only once."""
    return MedicalTriageAgent()

def main():
    st.set_page_config(page_title="MedAssist AI", layout="wide", page_icon="🧠")

    # ─── Premium Dark Theme CSS with Animations ───────────────────────────
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif !important;
        }

        .stApp {
            background-color: #0a0e17 !important;
            color: #E6EDF3 !important;
            background-image: 
                radial-gradient(ellipse at 10% 20%, rgba(59, 130, 246, 0.08), transparent 50%),
                radial-gradient(ellipse at 90% 80%, rgba(139, 92, 246, 0.06), transparent 50%),
                radial-gradient(ellipse at 50% 50%, rgba(6, 182, 212, 0.04), transparent 60%);
        }

        /* ─── Hero Animations ─────────────────────────────────── */
        @keyframes fadeDown {
            from { opacity: 0; transform: translateY(-25px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(25px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        @keyframes shimmer {
            0% { background-position: -200% center; }
            100% { background-position: 200% center; }
        }
        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-6px); }
        }

        .hero-title {
            font-size: 3.8rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 0px;
            background: linear-gradient(135deg, #06b6d4, #3b82f6, #8b5cf6, #06b6d4);
            background-size: 200% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: fadeDown 0.8s ease-out, shimmer 4s linear infinite;
        }
        
        .hero-badge {
            text-align: center;
            margin-bottom: 8px;
            animation: fadeDown 1s ease-out;
        }
        .hero-badge span {
            background: rgba(59, 130, 246, 0.15);
            border: 1px solid rgba(59, 130, 246, 0.3);
            color: #60a5fa;
            padding: 4px 14px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 500;
            letter-spacing: 1.5px;
            text-transform: uppercase;
        }
        
        .hero-subtitle {
            text-align: center;
            font-weight: 300;
            color: #6b7280;
            margin-bottom: 35px;
            font-size: 1.1rem;
            animation: fadeUp 1.2s ease-out;
        }

        /* ─── Triage Pulse Badges ─────────────────────────────── */
        @keyframes pulseHigh {
            0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.6); }
            70% { box-shadow: 0 0 0 18px rgba(239, 68, 68, 0); }
            100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
        }
        @keyframes pulseModerate {
            0% { box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.6); }
            70% { box-shadow: 0 0 0 18px rgba(245, 158, 11, 0); }
            100% { box-shadow: 0 0 0 0 rgba(245, 158, 11, 0); }
        }
        @keyframes pulseLow {
            0% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.6); }
            70% { box-shadow: 0 0 0 18px rgba(34, 197, 94, 0); }
            100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
        }

        .triage-High {
            background: linear-gradient(135deg, #ef4444, #dc2626);
            color: white; padding: 10px 28px; border-radius: 50px;
            font-weight: 600; font-size: 1.1rem; text-align: center;
            animation: pulseHigh 2s infinite; display: inline-block;
            box-shadow: 0 8px 24px rgba(239, 68, 68, 0.3);
        }
        .triage-Moderate {
            background: linear-gradient(135deg, #f59e0b, #d97706);
            color: #1a1a2e; padding: 10px 28px; border-radius: 50px;
            font-weight: 600; font-size: 1.1rem; text-align: center;
            animation: pulseModerate 2s infinite; display: inline-block;
            box-shadow: 0 8px 24px rgba(245, 158, 11, 0.3);
        }
        .triage-Low {
            background: linear-gradient(135deg, #22c55e, #16a34a);
            color: white; padding: 10px 28px; border-radius: 50px;
            font-weight: 600; font-size: 1.1rem; text-align: center;
            animation: pulseLow 2s infinite; display: inline-block;
            box-shadow: 0 8px 24px rgba(34, 197, 94, 0.3);
        }

        /* ─── Glass Cards ─────────────────────────────────────── */
        .glass-card {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.06);
            backdrop-filter: blur(20px);
            border-radius: 16px;
            padding: 24px;
            margin: 12px 0;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
            transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
            animation: fadeUp 0.8s ease-out;
        }
        .glass-card:hover {
            transform: translateY(-6px);
            border-color: rgba(59, 130, 246, 0.25);
            box-shadow: 0 12px 40px rgba(59, 130, 246, 0.1);
        }
        
        .glass-card-rank {
            display: inline-block;
            background: rgba(59, 130, 246, 0.15);
            color: #60a5fa;
            padding: 2px 10px;
            border-radius: 8px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-bottom: 8px;
        }

        /* ─── Confidence Bar ──────────────────────────────────── */
        .conf-bar-bg {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            height: 10px;
            margin-top: 10px;
            overflow: hidden;
        }
        .conf-bar-fill {
            height: 100%;
            border-radius: 10px;
            background: linear-gradient(90deg, #06b6d4, #3b82f6, #8b5cf6);
            transition: width 1.5s cubic-bezier(0.4, 0, 0.2, 1);
        }

        /* ─── Reasoning & Action Boxes ────────────────────────── */
        .reasoning-box {
            background: rgba(6, 182, 212, 0.04);
            border-left: 3px solid #06b6d4;
            padding: 18px 22px;
            border-radius: 0 12px 12px 0;
            color: #94a3b8;
            margin: 15px 0;
            line-height: 1.7;
            animation: fadeIn 1s ease-out;
        }
        
        .action-box {
            background: rgba(245, 158, 11, 0.04);
            border-left: 3px solid #f59e0b;
            padding: 18px 22px;
            border-radius: 0 12px 12px 0;
            color: #fbbf24;
            margin: 15px 0;
            font-weight: 500;
            animation: fadeIn 1.2s ease-out;
        }
        
        .disclaimer-bar {
            background: rgba(239, 68, 68, 0.04);
            border: 1px solid rgba(239, 68, 68, 0.15);
            border-radius: 12px;
            padding: 14px 20px;
            text-align: center;
            color: #9ca3af;
            font-size: 0.82rem;
            font-style: italic;
            margin-top: 30px;
            animation: fadeIn 1.5s ease-out;
        }

        /* ─── Section Divider ─────────────────────────────────── */
        .section-label {
            color: #6b7280;
            font-size: 0.7rem;
            letter-spacing: 2px;
            text-transform: uppercase;
            font-weight: 600;
            margin: 25px 0 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # ─── Hero Section ─────────────────────────────────────────────────
    st.markdown('<div class="hero-badge"><span>Agentic AI · Hybrid CDSS</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">MedAssist AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Autonomous Clinical Triage Agent · Forward Chaining + Random Forest</div>', unsafe_allow_html=True)

    agent = load_agent()
    
    # ─── Symptom Collector ────────────────────────────────────────────
    demo_symptoms = [
        "high_fever", "cough", "fatigue", "loss_of_smell", "breathlessness",
        "skin_rash", "swelled_lymph_nodes", "headache", "muscle_pain",
        "blister", "loss_of_appetite", "weight_loss", "sweating", "blood_in_sputum",
        "chills", "nausea", "dizziness", "fast_heart_rate", "muscle_weakness",
        "weakness_in_limbs", "chest_pain", "phlegm", "joint_pain", "vomiting",
        "sunken_eyes", "dehydration", "diarrhoea", "itching", "continuous_sneezing",
        # Critical disease symptoms (ensures High urgency can be triggered)
        "pain_behind_the_eyes", "rusty_sputum", "constipation", "belly_pain",
        "yellowish_skin", "dark_urine", "abdominal_pain", "mild_fever"
    ]
    
    ux_overrides = {
        "loss_of_smell": "Loss of Taste / Smell",
        "high_fever": "High Fever / Extreme Temperature",
        "blister": "Blisters",
        "continuous_sneezing": "Continuous Sneezing",
        "diarrhoea": "Diarrhoea",
        "pain_behind_the_eyes": "Pain Behind the Eyes",
        "rusty_sputum": "Rusty Sputum",
        "belly_pain": "Belly Pain / Stomach Ache",
        "yellowish_skin": "Yellowish Skin",
        "dark_urine": "Dark Urine",
        "mild_fever": "Mild Fever"
    }
    
    pretty_symptoms_map = {}
    for sym in demo_symptoms:
        if sym in agent.ml_tool.symptom_list:
            pretty_symptoms_map[sym] = ux_overrides.get(sym, sym.replace("_", " ").title())
            
    reversed_map = {v: k for k, v in pretty_symptoms_map.items()}
    display_symptoms = sorted(list(pretty_symptoms_map.values()))

    st.markdown('<div class="section-label">Symptom Collector</div>', unsafe_allow_html=True)
    
    selected = st.multiselect(
        "Select all symptoms you are currently experiencing:",
        options=display_symptoms,
        help="Type to search. Select multiple symptoms for better accuracy."
    )

    if st.button("Run Agent Diagnosis 🧠", use_container_width=True):
        if not selected:
            st.warning("Please select at least one symptom.")
        else:
            with st.spinner("🔄 Agent is orchestrating tools..."):
                time.sleep(1.2)
                
                active_symptoms = [reversed_map[sym] for sym in selected]
                results = agent.run(active_symptoms)
                
                st.markdown("---")
                
                if not results["top_predictions"]:
                    st.info("The agent could not confidently map these symptoms to a known condition. Consider providing more specific symptoms.")
                else:
                    # ─── Triage Banner ────────────────────────────────────
                    triage = results["triage"]
                    triage_class = f"triage-{triage['level']}"
                    
                    st.markdown(f"<div style='text-align:center; margin-bottom: 5px;'><div class='{triage_class}'>⚡ Urgency: {triage['level']}</div></div>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align:center; color:#6b7280; margin-top:12px; font-size:1rem;'>{triage['message']}</p>", unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)

                    # ─── Top 3 Predictions ────────────────────────────────
                    st.markdown('<div class="section-label">Diagnosis Output — Top 3 Probable Conditions</div>', unsafe_allow_html=True)
                    
                    for i, pred in enumerate(results["top_predictions"]):
                        conf_pct = pred['confidence'] * 100
                        ml_pct = pred['ml_contribution'] * 100
                        rule_pct = pred['rule_contribution'] * 100
                        rank_label = ["Primary Match", "Secondary Match", "Tertiary Match"][i] if i < 3 else f"Match #{i+1}"
                        
                        # Key symptoms display
                        sym_html = ""
                        if pred['matched_symptoms']:
                            pretty_syms = [s.replace('_', ' ').title() for s in pred['matched_symptoms']]
                            sym_tags = ''.join([f'<span style="background:rgba(59,130,246,0.1); color:#60a5fa; padding:2px 8px; border-radius:6px; font-size:0.75rem; margin-right:4px;">{s}</span>' for s in pretty_syms])
                            sym_html = f'<div style="margin-top:10px;">{sym_tags}</div>'
                        
                        card_html = f"""
                        <div class="glass-card">
                            <div class="glass-card-rank">{rank_label}</div>
                            <h3 style="margin:4px 0 6px 0; color:#e2e8f0; font-size:1.6rem;">{pred['disease']}</h3>
                            <div style="font-size:1rem; font-weight:500; color:#94a3b8;">
                                Confidence: <span style="color:#3b82f6; font-weight:700;">{conf_pct:.1f}%</span>
                            </div>
                            <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{conf_pct}%;"></div></div>
                            <div style="font-size:0.82rem; color:#6b7280; margin-top:12px;">
                                🤖 ML Probability: {ml_pct:.1f}% &nbsp;&nbsp;·&nbsp;&nbsp; 📋 Rule Satisfaction: {rule_pct:.1f}%
                            </div>
                            {sym_html}
                        </div>
                        """
                        st.markdown(card_html, unsafe_allow_html=True)

                    # ─── Agent Reasoning ──────────────────────────────────
                    st.markdown('<div class="section-label">Agent Reasoning</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="reasoning-box">🔍 {results["agent_reasoning"]}</div>', unsafe_allow_html=True)
                    
                    # ─── Action Plan ──────────────────────────────────────
                    st.markdown('<div class="section-label">Recommended Action</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="action-box">⚡ {triage["action_plan"]}</div>', unsafe_allow_html=True)
                    
                    # ─── Disclaimer ───────────────────────────────────────
                    st.markdown(f'<div class="disclaimer-bar">⚠️ {results["disclaimer"]}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
