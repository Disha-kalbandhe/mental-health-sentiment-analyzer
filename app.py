import streamlit as st
import joblib
import eli5
from eli5.sklearn import explain_prediction
import plotly.graph_objects as go
import re
from bs4 import BeautifulSoup

# Load model and vectorizer
model = joblib.load("best_sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def predict_with_confidence(text, vectorizer, model):
    text_vector = vectorizer.transform([text])
    predicted_label = model.predict(text_vector)[0]
    probabilities = model.predict_proba(text_vector)[0]
    class_labels = list(model.classes_)
    confidence = {class_labels[i]: float(probabilities[i]) for i in range(len(class_labels))}
    return predicted_label, confidence

def explain_eli5(text):
    explanation = eli5.format_as_html(
        eli5.explain_prediction(
            model,
            text,
            vec=vectorizer,
            top=10,
            target_names=model.classes_
        )
    )
    return explanation

st.set_page_config(
    page_title="Mental Health Sentiment Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Montserrat', sans-serif !important;
}
body {
    background: linear-gradient(120deg, #181c24 0%, #23272f 100%) fixed;
}
body:before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    z-index: -1;
    background: linear-gradient(120deg, #23272f 0%, #4B8BBE 100%);
    opacity: 0.12;
    animation: gradientBG 8s ease-in-out infinite alternate;
}
@keyframes gradientBG {
    0% { filter: blur(0px); }
    100% { filter: blur(8px); }
}
.neu-card {
    background: #23272f;
    border-radius: 18px;
    box-shadow: 0 8px 32px 0 rgba(75,139,190,0.18), 0 1.5px 4px 0 #181c24;
    padding: 32px 28px;
    margin: 32px auto 24px auto;
    max-width: 900px;
}
.header-logo {
    display: flex;
    flex-direction: column;
    align-items: center;
    margin-top: 32px;
    margin-bottom: 16px;
}
.header-logo h1 {
    font-size: 3.2em;
    font-weight: 700;
    background: linear-gradient(90deg, #4B8BBE, #306998 80%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2em;
    letter-spacing: 1px;
}
.header-logo h4 {
    color: #4B8BBE;
    font-weight: 400;
    margin-top: 0;
    margin-bottom: 0.5em;
}
.stButton > button, .st-expander > label, .stDownloadButton > button {
    border-radius: 30px;
    background: linear-gradient(90deg, #4B8BBE, #306998 80%);
    border: none;
    color: white !important;
    padding: 16px 0;
    font-size: 1.13em;
    font-weight: bold;
    box-shadow: 0 0 16px #4B8BBE44, 0 0 0 0 #24F03F;
    transition: all 0.25s cubic-bezier(.25,.8,.25,1), box-shadow 0.4s cubic-bezier(.25,.8,.25,1);
    margin-top: 12px;
    letter-spacing: 1px;
    outline: none;
    position: relative;
    overflow: hidden;
}
.stButton > button:hover, .st-expander > label:hover, .stDownloadButton > button:hover {
    filter: brightness(1.18) drop-shadow(0 0 8px #24F03F99);
    box-shadow: 0 0 32px #4B8BBE99, 0 0 16px 4px #24F03F66;
    transform: translateY(-3px) scale(1.045) rotate(-1deg);
    border: 2.5px solid #24F03F;
    background: linear-gradient(90deg, #24F03F 0%, #4B8BBE 100%);
    color: #fff !important;
}
.stButton > button:active, .stDownloadButton > button:active {
    background: linear-gradient(90deg, #306998, #4B8BBE 80%);
    box-shadow: 0 0 8px #24F03F99;
    transform: scale(0.98);
}
.stButton > button:focus, .stDownloadButton > button:focus {
    outline: 2.5px solid #24F03F;
    box-shadow: 0 0 0 3px #24F03F55;
}
.stTextArea > div > div > textarea {
    background-color: #23272f;
    color: #fff;
    border: 1.5px solid #4B8BBE;
    border-radius: 12px;
    padding: 18px;
    font-size: 1.1em;
    box-shadow: 0 2px 12px #4B8BBE11;
}
.about-card {
    background: #23272f;
    border-radius: 16px;
    box-shadow: 0 4px 24px #4B8BBE22;
    padding: 28px 22px;
    margin-top: 18px;
    color: #bfc9d1;
}
h4, h3, h2 {
    color: #4B8BBE !important;
    font-weight: 700;
    margin-bottom: 0.7em;
}
.word-card {
    background: #23272f;
    border-radius: 16px;
    box-shadow: 0 4px 24px #4B8BBE22;
    padding: 28px 22px;
    margin-top: 18px;
    color: #bfc9d1;
}
.eli5-scroll {
    max-height: 350px;
    overflow-y: auto;
    background: #181c24;
    border-radius: 10px;
    padding: 12px 18px;
    margin-top: 10px;
    color: #fff;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='header-logo'>
    <img src='https://img.icons8.com/fluency/96/brain.png' width='80' style='margin-bottom: 10px;'>
    <h1>Mental Health Sentiment Analyzer</h1>
    <h4>Advanced NLP Analysis</h4>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='neu-card'>", unsafe_allow_html=True)
col1, col2 = st.columns([2, 1])
with col1:
    user_input = st.text_area(
        "📝 Enter a mental health-related post or message:",
        placeholder="e.g., I feel like giving up. Everything is so heavy.",
        height=180
    )
with col2:
    st.markdown("""
    <div class='about-card'>
        <h4>ℹ️ About</h4>
        <p>This analyzer uses advanced machine learning to assess mental health-related messages.<br>It can identify potential signs of concerning content while maintaining user privacy.</p>
    </div>
    """, unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

analyze_btn = st.button("🔍 Analyze Message", key="analyze_btn", help="Click to analyze the message")

if analyze_btn:
    if not user_input.strip():
        st.warning("⚠️ Please enter a message to analyze.")
    else:
        with st.spinner('Analyzing message...'):
            predicted_label, confidence_scores = predict_with_confidence(user_input, vectorizer, model)
        st.markdown("<div class='neu-card'>", unsafe_allow_html=True)
        if predicted_label.lower() == "suicidal":
            st.error("🔴 **Predicted Sentiment: SUICIDAL ⚠️**")
        else:
            st.success("🟢 **Predicted Sentiment: NON-SUICIDAL 😊**")
        confidence_percent = {label: prob * 100 for label, prob in confidence_scores.items()}
        st.markdown("<h4>Confidence Distribution</h4>", unsafe_allow_html=True)
        pie_colors = ['#EE1515', '#24F03F'] if 'suicidal' in confidence_percent else ['#4B8BBE', '#24F03F']
        pie_labels = list(confidence_percent.keys())
        pie_values = list(confidence_percent.values())
        pie_fig = go.Figure(data=[go.Pie(
            labels=pie_labels,
            values=pie_values,
            hole=0.35,
            marker=dict(colors=pie_colors,
                        line=dict(color='#181c24', width=3)),
            textinfo='label+percent',
            insidetextorientation='radial',
            pull=[0.08 if v == max(pie_values) else 0 for v in pie_values],
            hoverinfo='label+percent+value',
            sort=False,
            hovertemplate='<b>%{label}</b><br>Confidence: <span style="color:#4B8BBE;font-size:1.2em;">%{percent}</span><br>Score: <span style="color:#24F03F;font-size:1.1em;">%{value:.2f}%</span><extra></extra>',
            rotation=120,
            opacity=0.96
        )])
        pie_fig.update_traces(
            textfont_size=20,
            marker=dict(
                line=dict(color='#23272f', width=3),
                colors=pie_colors
            ),
            pull=[0.08 if v == max(pie_values) else 0 for v in pie_values],
            opacity=0.96,
            automargin=True
        )
        pie_fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#fff',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5,
                font=dict(color="#bfc9d1", size=16)
            ),
            margin=dict(l=20, r=20, t=30, b=20),
            transition={'duration': 700, 'easing': 'cubic-in-out'}
        )
        st.plotly_chart(pie_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        with st.expander("🔍 Word Importance Analysis (Click to Expand)", expanded=True):
            st.markdown("""
            <style>
            .custom-eli5-table {
                width: 100%;
                border-collapse: separate;
                border-spacing: 0;
                background: #23272f;
                border-radius: 14px;
                box-shadow: 0 4px 24px #4B8BBE22;
                margin-bottom: 18px;
                font-size: 1.08em;
                overflow: hidden;
            }
            .custom-eli5-table th {
                background: #4B8BBE;
                color: #fff;
                font-weight: 700;
                padding: 12px 16px;
                border-bottom: 2px solid #306998;
                text-align: center;
            }
            .custom-eli5-table td {
                color: #e3e9f0;
                padding: 10px 16px;
                border-bottom: 1px solid #23272f;
                text-align: center;
            }
            .custom-eli5-table tr:last-child td {
                border-bottom: none;
            }
            .custom-eli5-table .pos {
                background: #1e2f23;
                color: #24F03F;
                font-weight: bold;
            }
            .custom-eli5-table .neg {
                background: #2f1e1e;
                color: #EE1515;
                font-weight: bold;
            }
            .custom-eli5-table .feature {
                font-family: 'Montserrat', monospace;
                font-size: 1.08em;
            }
            .custom-eli5-table .icon {
                font-size: 1.2em;
                vertical-align: middle;
                margin-right: 6px;
            }
            </style>
            <div class='eli5-beauty-card'>
                <h4>How did the model decide?</h4>
                <p>The following table highlights the most influential words in your message that contributed to the prediction. <span style='color:#24F03F;'>🟢</span> = positive, <span style='color:#EE1515;'>🔴</span> = negative contribution.</p>
            </div>
            """, unsafe_allow_html=True)
            try:
                explanation = explain_eli5(user_input)
                table_match = re.search(r"(<table[\s\S]*?</table>)", explanation)
                table_html = table_match.group(1) if table_match else ""
                if table_html:
                    soup = BeautifulSoup(table_html, "html.parser")
                    headers = [th.get_text(strip=True) for th in soup.find_all('th')]
                    rows = []
                    for tr in soup.find_all('tr')[1:]:
                        tds = tr.find_all('td')
                        if len(tds) == 2:
                            contrib = tds[0].get_text(strip=True)
                            feature = tds[1].get_text(strip=True)
                            try:
                                val = float(contrib.replace('+','').replace(',',''))
                                if val > 0:
                                    contrib_html = f"<span class='icon'>🟢</span><span class='pos'>{contrib}</span>"
                                else:
                                    contrib_html = f"<span class='icon'>🔴</span><span class='neg'>{contrib}</span>"
                            except:
                                contrib_html = contrib
                            rows.append(f"<tr><td>{contrib_html}</td><td class='feature'>{feature}</td></tr>")
                    custom_table = f"""
                    <table class='custom-eli5-table'>
                        <thead><tr><th>{headers[0]}</th><th>{headers[1]}</th></tr></thead>
                        <tbody>{''.join(rows)}</tbody>
                    </table>
                    """
                    st.markdown(custom_table, unsafe_allow_html=True)
            except Exception as e:
                st.error("Could not generate word importance visualization.")
        st.write("---")
        if predicted_label.lower() == "suicidal":
            st.error("⚠️ If you're struggling, please reach out for help. You're not alone. ❤️")
        else:
            st.success("✅ Keep taking care of your mental well-being. You're doing great! 💚")
st.write("---")
st.markdown("<div style='text-align: center; color: #4B8BBE; font-size: 1.1em;'>Made by Disha Kalbandhe | AI/ML Enthusiast</div>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: #bfc9d1;'>Model Accuracy: 94% | Logistic Regression + TF-IDF</div>", unsafe_allow_html=True)
