#AIzaSyAdoKGT8c8SMaikKeTnYkywyVvb0XWcI4U


import os
import pickle
import hashlib
from typing import List, Dict, Optional
import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from PyPDF2 import PdfReader
import time
from datetime import datetime, timedelta

# Konfigürasyon
class Config:
    GEMINI_API_KEY = "AIzaSyAdoKGT8c8SMaikKeTnYkywyVvb0XWcI4U"  # Buraya API anahtarınızı ekleyin
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    CHUNK_SIZE = 700 
    CHUNK_OVERLAP = 70 
    CACHE_DURATION_HOURS = 24
    MAX_CONTEXT_LENGTH = 10000  # Gemini prompt limiti için

class PDFChatbot:
    def __init__(self):
        self.embedding_model = None
        self.vector_store = None
        self.chunks = []
        self.cache = {}
        self.cache_file = "chatbot_cache.pkl"
        self.load_cache()
        
        # Gemini yapılandırması
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
    def load_cache(self):
        """Cache dosyasını yükle"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                # Eski cache'leri temizle
                current_time = datetime.now()
                self.cache = {
                    k: v for k, v in self.cache.items() 
                    if current_time - v.get('timestamp', datetime.min) < timedelta(hours=Config.CACHE_DURATION_HOURS)
                }
        except Exception as e:
            print(f"Cache yükleme hatası: {e}")
            self.cache = {}
    
    def save_cache(self):
        """Cache'i dosyaya kaydet"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            print(f"Cache kaydetme hatası: {e}")
    
    def get_pdf_hash(self, pdf_content: bytes) -> str:
        """PDF içeriğinin hash'ini al"""
        return hashlib.md5(pdf_content).hexdigest()
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """PDF'den metin çıkar"""
        try:
            pdf_reader = PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"PDF okuma hatası: {e}")
            return ""
    
    def create_chunks(self, text: str) -> List[str]:
        """Metni parçalara böl"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), Config.CHUNK_SIZE - Config.CHUNK_OVERLAP):
            chunk = " ".join(words[i:i + Config.CHUNK_SIZE])
            if len(chunk.strip()) > 5:  # Çok kısa parçaları filtrele
                chunks.append(chunk.strip())
        
        return chunks
    
    def initialize_embeddings(self):
        """Embedding modelini başlat"""
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
    
    def create_vector_store(self, chunks: List[str]) -> faiss.IndexFlatIP:
        """Vektör veritabanı oluştur"""
        self.initialize_embeddings()
        
        # Embedding'leri oluştur
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=True)
        embeddings = embeddings.astype('float32')
        
        # FAISS index oluştur
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        
        # Normalize et (cosine similarity için)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        
        return index
    
    def process_pdf(self, pdf_file):
        """PDF'i işle ve vektör veritabanını oluştur"""
        # PDF hash'ini kontrol et
        pdf_content = pdf_file.read()
        pdf_file.seek(0)  # Dosya pointer'ını başa al
        pdf_hash = self.get_pdf_hash(pdf_content)
        
        # Cache'de var mı kontrol et
        if pdf_hash in self.cache:
            cached_data = self.cache[pdf_hash]
            self.chunks = cached_data['chunks']
            self.vector_store = cached_data['vector_store']
            st.success("PDF cache'den yüklendi!")
            return True
        
        # PDF'i işle
        with st.spinner("PDF işleniyor..."):
            text = self.extract_text_from_pdf(pdf_file)
            
            if not text.strip():
                st.error("PDF'den metin çıkarılamadı!")
                return False
            
            # Parçalara böl
            self.chunks = self.create_chunks(text)
            
            if not self.chunks:
                st.error("Geçerli metin parçası bulunamadı!")
                return False
            
            # Vektör veritabanını oluştur
            self.vector_store = self.create_vector_store(self.chunks)
            
            # Cache'e kaydet
            self.cache[pdf_hash] = {
                'chunks': self.chunks,
                'vector_store': self.vector_store,
                'timestamp': datetime.now()
            }
            self.save_cache()
            
            st.success(f"PDF başarıyla işlendi! {len(self.chunks)} parça oluşturuldu.")
            return True
    
    def semantic_search(self, query: str, k: int = 5) -> List[str]:
        """Semantic search yap"""
        if self.vector_store is None or not self.chunks:
            return []
        
        self.initialize_embeddings()
        
        # Query embedding'i oluştur
        query_embedding = self.embedding_model.encode([query]).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Arama yap
        scores, indices = self.vector_store.search(query_embedding, k)
        
        # Sonuçları döndür
        results = []
        for i, score in zip(indices[0], scores[0]):
            if score > 0.3:  # Minimum benzerlik eşiği
                results.append(self.chunks[i])
        
        return results
    
    def generate_response(self, question: str, context: List[str]) -> str:
        """Gemini ile cevap üret"""
        # Context'i birleştir ve kısalt
        combined_context = "\n\n".join(context)
        if len(combined_context) > Config.MAX_CONTEXT_LENGTH:
            combined_context = combined_context[:Config.MAX_CONTEXT_LENGTH] + "..."
        
        # Cache kontrolü
        cache_key = hashlib.md5(f"{question}_{combined_context}".encode()).hexdigest()
        if cache_key in self.cache:
            cached_response = self.cache[cache_key]
            if datetime.now() - cached_response['timestamp'] < timedelta(hours=1):
                return cached_response['response']
        
        prompt = f"""
        Aşağıdaki belgelerden elde edilen bilgilere dayanarak soruyu cevapla.
        Cevabın doğru, net ve faydalı olmalı. Eğer belgede bilgi yoksa, bunu belirt.
        
        BELGELER:
        {combined_context}
        
        SORU: {question}
        
        CEVAP:
        """
        
        try:
            response = self.model.generate_content(prompt)
            generated_response = response.text
            
            # Cache'e kaydet
            self.cache[cache_key] = {
                'response': generated_response,
                'timestamp': datetime.now()
            }
            self.save_cache()
            
            return generated_response
            
        except Exception as e:
            return f"Cevap üretilirken hata oluştu: {str(e)}"
    
    def chat(self, question: str) -> str:
        """Ana chat fonksiyonu"""
        if not self.vector_store or not self.chunks:
            return "Lütfen önce bir PDF dosyası yükleyin."
        
        # Semantic search
        relevant_chunks = self.semantic_search(question, k=15)
        
        if not relevant_chunks:
            return "Bu konuyla ilgili belgede bilgi bulunamadı."
        
        # Cevap üret
        response = self.generate_response(question, relevant_chunks)
        return response

def main():
    st.set_page_config(
        page_title="PDF Chatbot",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 PDF Chatbot")
    st.markdown("PDF belgelerinizi yükleyin ve içeriği hakkında sorular sorun!")
    
    # Chatbot'u session state'de sakla
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = PDFChatbot()
    
    # Sidebar - PDF yükleme
    with st.sidebar:
        st.header("📄 PDF Yükle")
        uploaded_file = st.file_uploader(
            "PDF dosyanızı seçin",
            type=['pdf'],
            help="PDF dosyanızı buraya sürükleyin veya seçin"
        )
        
        if uploaded_file is not None:
            if st.button("PDF'i İşle", type="primary"):
                success = st.session_state.chatbot.process_pdf(uploaded_file)
                if success:
                    st.session_state.pdf_processed = True
    
    # Ana alan - Chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Sohbet")
        
        # Chat geçmişi
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            for i, (question, answer) in enumerate(st.session_state.chat_history):
                with st.chat_message("user"):
                    st.write(question)
                with st.chat_message("assistant"):
                    st.write(answer)
        
        # Soru sorma
        if hasattr(st.session_state, 'pdf_processed') and st.session_state.pdf_processed:
            question = st.chat_input("PDF'iniz hakkında bir soru sorun...")
            
            if question:
                with st.chat_message("user"):
                    st.write(question)
                
                with st.chat_message("assistant"):
                    with st.spinner("Cevap hazırlanıyor..."):
                        answer = st.session_state.chatbot.chat(question)
                    st.write(answer)
                
                # Chat geçmişine ekle
                st.session_state.chat_history.append((question, answer))
        else:
            st.info("Soru sormaya başlamak için lütfen bir PDF dosyası yükleyin ve işleyin.")
    
    with col2:
        st.header("ℹ️ Bilgi")
        
        if hasattr(st.session_state, 'pdf_processed') and st.session_state.pdf_processed:
            st.success("✅ PDF başarıyla yüklendi")
            st.info(f"📊 {len(st.session_state.chatbot.chunks)} metin parçası oluşturuldu")
        else:
            st.warning("⏳ PDF bekleniyor")
        
        st.markdown("---")
        st.markdown("**💡 Kullanım İpuçları:**")
        st.markdown("• Açık ve spesifik sorular sorun")
        st.markdown("• PDF'inizin dilinde soru sorun")
        st.markdown("• Uzun belgeler için sabırlı olun")
        
        # Cache temizleme
        if st.button("🗑️ Cache Temizle"):
            if os.path.exists(st.session_state.chatbot.cache_file):
                os.remove(st.session_state.chatbot.cache_file)
            st.session_state.chatbot.cache = {}
            st.success("Cache temizlendi!")

if __name__ == "__main__":
    main()