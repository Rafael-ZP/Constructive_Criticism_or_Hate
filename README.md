# 🎬 Constructive Criticism or Hate?  
### An NLP-Based Movie Review Classification System Using DeBERTa

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.9-blue)
![Model](https://img.shields.io/badge/model-DeBERTa-v3--base-orange)

## 📌 Abstract

In the age of digital media, movie reviews hold significant influence over public perception. However, most sentiment analysis systems broadly classify reviews as positive, neutral, or negative—failing to identify whether a negative review is **constructive feedback** or **hate speech**. This project fine-tunes **DeBERTa (Decoding-enhanced BERT with Disentangled Attention)** to address this gap.

The resulting model is integrated into a **Streamlit web interface** for real-time classification and visualization. The system improves AI-based moderation by providing **deeper semantic analysis**, outperforming traditional BERT-based approaches.

---

## 📚 Features

- ✅ Fine-tuned DeBERTa for binary classification  
- ✅ Real-time text classification using Streamlit  
- ✅ Metrics: Accuracy, Precision, Recall, F1 Score  
- ✅ Confusion matrix and performance graphs  
- ✅ Model & tokenizer saving for future predictions  

---

## ⚙️ Methodology

1. **Data Collection & Preprocessing**  
   - 400 labeled reviews (200 Constructive Criticism, 200 Hate Speech)  
   - Cleaning, lemmatization, and tokenization  

2. **Model Training**  
   - Model: `microsoft/deberta-v3-base`  
   - Optimizer: AdamW  
   - Loss Function: CrossEntropy  
   - Evaluation Metrics: Accuracy, F1 Score  

3. **Model Evaluation**  
   - Confusion matrix  
   - Epoch-wise Accuracy & F1-score plots  

4. **Deployment**  
   - Streamlit app for real-time user input and classification

---

## 📊 Results

| Model    | Accuracy (%) | Precision (%) | Recall (%) | F1 Score (%) |
|----------|--------------|----------------|--------------|---------------|
| BERT     | 85.2         | 83.5           | 86.1         | 84.8          |
| DeBERTa  | **91.4**     | **89.7**       | **92.2**     | **90.9**      |

- ✔️ Fewer false positives & negatives  
- ✔️ Better understanding of nuanced expressions  
- ✔️ High classification accuracy even on implicit hate speech  

---

## 🧠 Tech Stack

- **Model**: DeBERTa-v3-base (`transformers`)
- **Language**: Python 3.9+
- **Frontend**: Streamlit
- **Training**: PyTorch + Hugging Face Trainer

---

## 🚀 How to Run

bash
# Clone this repository
git clone https://github.com/yourusername/movie-review-criticism-classifier

# Create a virtual environment & activate
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py



⸻

📸 Output Screenshots

🖼️ Add screenshots here of your Streamlit UI, confusion matrix, accuracy chart, etc.

⸻

🌍 Impact on Society
	•	Promotes respectful digital conversations
	•	Reduces workload for human content moderators
	•	Encourages ethical review culture
	•	Aids platforms in maintaining a healthier online environment

⸻

🔭 Future Scope
	•	Expansion to multilingual reviews
	•	Adaptation for social media platforms
	•	Integration of explainable AI techniques
	•	Dataset scaling for robustness

⸻

📄 License

This project is licensed under the MIT License.

⸻

🤝 Contributors
	•	👤 [Your Name]
	•	📧 [Your Email or GitHub]

⸻

📢 Acknowledgments

Special thanks to Hugging Face, Streamlit, and the open-source NLP community for toolkits and pretrained models.

---

Let me know if you'd like this split into separate files (`app.py`, `requirements.txt`, etc.), or want badges and links tailored to your GitHub username.
