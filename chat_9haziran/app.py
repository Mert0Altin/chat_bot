from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from chat_module import PDFChatbot  # chat_module.py dosyasına göre
import os

app = Flask(__name__)
CORS(app)

# Chatbot nesnesini başlat (ve PDF'i yükle)
chatbot = PDFChatbot()
pdf_yüklendi = chatbot.load_and_process_pdf()

@app.route('/')
def home():
    return render_template("chatbot.html")  # HTML arayüzünü döndür

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({'answer': "Lütfen geçerli bir soru girin."})

    if not pdf_yüklendi:
        return jsonify({'answer': "PDF yüklenemedi. Lütfen sistem yöneticisine danışın."})

    answer = chatbot.chat(question)
    return jsonify({'answer': answer})

if __name__ == "__main__":
    app.run(debug=True)
