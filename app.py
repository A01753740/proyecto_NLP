from utils.install_requirements import install_requirements
install_requirements()
from flask import Flask, render_template, request, jsonify
from functions.preprocess import load_data, preprocess_data
from agents.sentiment_analysis import analyze_sentiment

app = Flask(__name__)

@app.route("/")
def home():
    """Renderiza la página principal."""
    return render_template("index.html")  

@app.route("/analyze_feedback", methods=["POST"])
def analyze_feedback():
    """Analiza el sentimiento de un texto utilizando LangChain con RAG."""
    feedback = request.json.get("feedback", "")
    print(f"Texto recibido: {feedback}")  

    if not feedback:
        return jsonify({"error": "No feedback provided"}), 400

    result = analyze_sentiment(feedback)
    print(f"Resultado del análisis: {result}") 
    return jsonify({"feedback": feedback, "analysis": result})

if __name__ == "__main__":
    app.run(debug=True)