# 🧠 Mental Health Sentiment Analyzer

A powerful yet lightweight machine learning web app that detects **suicidal** vs **non-suicidal** sentiment in mental health-related messages using **Logistic Regression + TF-IDF**.

> 🔗 **Live App**: [Click here to try the app](https://mental-health-sentiment-analyzer-bkaqn7vkqbxhexj2eescms.streamlit.app/)

---

## 💡 Why This Project?

Mental health struggles are often expressed online — through tweets, posts, or messages. This project aims to:

- Detect early signs of suicidal ideation 💬⚠️
- Offer real-time, explainable predictions for mental health posts
- Provide a deployable, lightweight ML solution with interpretability

---

## 🚀 Features

✅ Predicts sentiment as either **Suicidal ⚠️** or **Non-Suicidal 😺✨**  
✅ Shows **confidence score** and a beautiful **pie chart** of prediction confidence  
✅ Built-in **interpretability** with ELI5 so users can understand *why* the model made a prediction  
✅ Clean, friendly UI using **Streamlit**  
✅ 94% accuracy on test data 🧪

---

## 🧠 Model Overview

- **Model**: Logistic Regression
- **Vectorization**: TF-IDF (trained on merged suicide-related datasets)
- **Accuracy**: ~94% on the test set
- **Tools**: `scikit-learn`, `matplotlib`, `eli5`, `streamlit`

---

## 🖥️ Run Locally

Clone the repo and run it locally:

```bash
git clone https://github.com/Disha-kalbandhe/mental-health-sentiment-analyzer.git
cd mental-health-sentiment-analyzer
pip install -r requirements.txt
streamlit run app.py

📂 Project Structure

📦 mental-health-sentiment-analyzer
├── app.py                         ← Streamlit app
├── requirements.txt              ← Required dependencies
├── data/
│   ├── tfidf_vectorizer.pkl      ← TF-IDF vectorizer
│   ├── best_sentiment_model.pkl  ← Trained logistic regression model
├── step_1_merge_clean.py         ← Dataset merging + cleaning script
├── step_2_train_model.py         ← Model training + saving script
├── test_input.py                 ← Sample test input script

🔍 Example Inputs
| Input Text                                 | Prediction              | Confidence |
| ------------------------------------------ | --------------------    | ---------- |
| *"I feel like giving up on everything."*   | **Suicidal ⚠️**        | 97.6%      |
| *"I’m feeling much better after therapy."* | **Non-Suicidal 😺✨**  | 94.3%      |


👤 Author
Disha Kalbandhe
🎓 B.Tech Electronics & Computer Science
🌱 AI/ML Enthusiast
📧 LinkedIn
https://www.linkedin.com/in/disha-kalbandhe-831b78278

⭐️ Show Your Support
If you liked this project:
Star ⭐ this repo
Share 💬 it with peers
Connect on LinkedIn



