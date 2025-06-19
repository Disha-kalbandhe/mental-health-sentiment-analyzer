import streamlit as st
import streamlit.components.v1 as components
from streamlit.components.v1 import html
import joblib
import eli5
from eli5.sklearn import explain_prediction
import matplotlib.pyplot as plt

# Load model and vectorizer
model = joblib.load("data/best_sentiment_model.pkl")
vectorizer = joblib.load("data/tfidf_vectorizer.pkl")

# --------- Prediction Function ----------
def predict_with_confidence(text, vectorizer, model):
    text_vector = vectorizer.transform([text])
    predicted_label = model.predict(text_vector)[0]
    probabilities = model.predict_proba(text_vector)[0]
    class_labels = list(model.classes_)
    confidence = {class_labels[i]: float(probabilities[i]) for i in range(len(class_labels))}
    return predicted_label, confidence

# --------- ELI5 Explanation (Handled safely) ----------
def explain_eli5(text):
    explanation = eli5.format_as_html(
        eli5.explain_prediction(
            model,
            text,  # 🚨 Pass the raw string, NOT the vectorized form!
            vec=vectorizer,
            top=10,
            target_names=model.classes_
        )
    )
    return explanation


# --------- Streamlit Page Config & Styling ----------
st.set_page_config(page_title="Mental Health Sentiment Analyzer", layout="centered")

# Custom CSS
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.reportview-container {padding: 2rem;}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🧠 Mental Health Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Logistic Regression + TF-IDF</h4>", unsafe_allow_html=True)
st.markdown("💬 This app predicts whether a mental health message shows **suicidal** or **non-suicidal** sentiment.")

st.write("---")

# Text input
user_input = st.text_area("📝 Enter a mental health-related post or message:",
    placeholder="e.g., I feel like giving up. Everything is so heavy.",
    height=200
)

# On button click
if st.button("🔍 Analyze"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a message to analyze.")
    else:
        predicted_label, confidence_scores = predict_with_confidence(user_input, vectorizer, model)

        # Display predicted result
        if predicted_label.lower() == "suicidal":
            st.error("🔴 **Predicted Sentiment: SUICIDAL ⚠️**")
        else:
            st.success("🟢 **Predicted Sentiment: NON-SUICIDAL 😺✨**")

        # Convert to percentage
        confidence_percent = {label: prob * 100 for label, prob in confidence_scores.items()}

        # Show confidence scores
        st.subheader("📊 Confidence Scores:")
        for label, score in confidence_percent.items():
            emoji = "⚠️" if label.lower() == "suicidal" else "😺✨"
            st.markdown(f"🔹 **{label.capitalize()} {emoji}**: `{score:.2f}%`")

        # Pie chart
        st.subheader("📈 Confidence Distribution:")
        fig, ax = plt.subplots()
        colors = [ "#24F03F","#EE1515"]
        ax.pie(confidence_percent.values(), labels=confidence_percent.keys(), autopct='%1.1f%%', startangle=140, colors=colors)
        ax.axis('equal')
        st.pyplot(fig)

        # Interpretability
        st.subheader("🧠 Why this prediction?")
        try:
            # Nuclear CSS option
            st.markdown("""
            <style>
                /* Nuclear option - target all tables and cells */
                .eli5 table, 
                .eli5 table th, 
                .eli5 table td,
                .eli5 tr,
                .eli5 tbody,
                .eli5 thead {
                    background-color: #252525 !important;
                    color: white !important;
                }
                
                /* Add colored borders for positive/negative contributions */
                .eli5 .pos {
                    border: 1px solid #4CAF50 !important; /* Green for positive */
                    box-shadow: 0 0 5px rgba(76, 175, 80, 0.3);
                }
                
                .eli5 .neg {
                    border: 1px solid #F44336 !important; /* Red for negative */
                    box-shadow: 0 0 5px rgba(244, 67, 54, 0.3);
                }
                
                /* Override any element with background */
                .eli5 * {
                    background-color: transparent !important;
                    color: white !important;
                }
            </style>
            """, unsafe_allow_html=True)
            
            explanation_html = explain_eli5(user_input)
            
            # JavaScript-enforced solution as backup
            wrapped_html = f"""
            <div id="eli5-container" style="background-color: #1e1e1e; padding: 15px; border-radius: 8px;">
                {explanation_html}
                <script>
                // Nuclear JavaScript solution
                function forceStyles() {{
                    // Target all tables
                    var tables = document.querySelectorAll('table');
                    tables.forEach(function(table) {{
                        table.style.backgroundColor = '#252525';
                        table.style.color = 'white';
                        
                        // Style all cells
                        var cells = table.querySelectorAll('th, td');
                        cells.forEach(function(cell) {{
                            cell.style.backgroundColor = '#252525';
                            cell.style.color = 'white';
                            
                            // Add colored borders based on content
                            if (cell.textContent.includes('+')) {{
                                cell.style.border = '1px solid #4CAF50';
                                cell.style.boxShadow = '0 0 5px rgba(76, 175, 80, 0.3)';
                            }} else if (cell.textContent.includes('-')) {{
                                cell.style.border = '1px solid #F44336';
                                cell.style.boxShadow = '0 0 5px rgba(244, 67, 54, 0.3)';
                            }}
                        }});
                    }});
                    
                    // Force all text white
                    var allElements = document.querySelectorAll('#eli5-container *');
                    allElements.forEach(function(el) {{
                        el.style.color = 'white';
                    }});
                }}
                
                // Run immediately and every 200ms for 1 second to catch dynamic elements
                forceStyles();
                var interval = setInterval(forceStyles, 200);
                setTimeout(function() {{ clearInterval(interval); }}, 1000);
                </script>
            </div>
            """
            
            components.html(wrapped_html, height=500, scrolling=True)
            
        except Exception as e:
            st.warning(f"⚠️ Interpretability not available. Error: {str(e)}")

        # Friendly message
        st.write("---")
        if predicted_label.lower() == "suicidal":
            st.error("⚠️ If you're struggling, please reach out for help. You're not alone. ❤️")
        
        else:
            st.success("✅ Keep taking care of your mental well-being. You're doing great! 💚")

# Footer
st.write("---")
st.markdown("<div style='text-align: center;'>Made by Disha Kalbandhe | AI/ML Enthusiast</div>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center;'>Model Accuracy: 94% | Logistic Regression + TF-IDF</div>", unsafe_allow_html=True)
