# 🎮 NumPy-100 Transformer Game

Welcome to the **NumPy-100 Transformer Game** — an interactive Streamlit app that combines coding practice with Transformer-powered attention visualization!

✅ Solve NumPy-100-style problems  
✅ Write and test your own NumPy code  
✅ Feed your outputs through a custom Transformer built with pure NumPy  
✅ Visualize and compare attention maps between your solution and the correct one  
✅ Get instant feedback on code correctness and attention similarity  

---

## 🚀 Features
- **100 NumPy problems** (or as many as you load)
- Real-time code execution with output comparison
- Custom Transformer (NumpyFormerEncoder) for attention visualization
- Side-by-side heatmaps of attention heads (your output vs expected)
- Automatic similarity scoring (MSE of attention patterns)
- Friendly UI with Streamlit

---

## 📦 Project structure
numpy_game/
├── app.py # Streamlit app
├── numpyformer/
│ ├── encoder.py # NumpyFormer: positional encoding + attention
│ ├── positional.py
│ ├── utils.py
├── exercises/
│ └── problems.py # Q&A pairs
├── requirements.txt
└── README.md

yaml
Copy
Edit

---

## ⚡ Installation

Clone the repo:
```bash
git clone https://github.com/your-username/numpy-transformer-game.git
cd numpy-transformer-game
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
🏁 Run the app
bash
Copy
Edit
streamlit run app.py
Then open your browser at http://localhost:8501

🎨 How it works
1️⃣ Select a NumPy problem
2️⃣ Write your NumPy solution code
3️⃣ The app runs your code + reference code
4️⃣ Both outputs pass through a Transformer (pure NumPy attention)
5️⃣ View side-by-side attention heatmaps
6️⃣ See similarity score + feedback

