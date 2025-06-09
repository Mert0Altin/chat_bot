import os
import pickle
import hashlib
from typing import List, Dict, Optional, Set 
import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from PyPDF2 import PdfReader
import time
from datetime import datetime, timedelta
import re

# Konfigürasyon
class Config:
    GEMINI_API_KEY = "AIzaSyAdoKGT8c8SMaikKeTnYkywyVvb0XWcI4U"  # Buraya API anahtarınızı ekleyin
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    CHUNK_SIZE = 300  # Daha küçük chunk'lar
    CHUNK_OVERLAP = 75  # Daha fazla örtüşme
    CACHE_DURATION_HOURS = 24
    MAX_CONTEXT_LENGTH = 6000  # Gemini prompt limiti için
    PDF_FILE_PATH = "document.pdf"  # Sabit PDF dosyası yolu

class PDFChatbot:
    def __init__(self):
        self.embedding_model = None
        self.vector_store = None
        self.chunks = []
        self.cache = {}
        self.cache_file = "chatbot_cache.pkl"
        self.pdf_processed = False
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
    
    def get_pdf_hash(self, pdf_path: str) -> str:
        """PDF dosyasının hash'ini al"""
        try:
            with open(pdf_path, 'rb') as f:
                pdf_content = f.read()
            return hashlib.md5(pdf_content).hexdigest()
        except Exception as e:
            print(f"PDF hash alma hatası: {e}")
            return ""
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """PDF'den metin çıkar - gelişmiş ve kapsamlı metin çıkarma"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                text = ""
                
                # PDF metadata bilgilerini al
                metadata = pdf_reader.metadata
                if metadata:
                    text += "=== PDF BİLGİLERİ ===\n"
                    for key, value in metadata.items():
                        if value:
                            text += f"{key}: {value}\n"
                    text += "\n"
                
                # Her sayfayı detaylı işle
                for page_num, page in enumerate(pdf_reader.pages):
                    # Sayfa başlığı
                    text += f"\n=== SAYFA {page_num + 1} ===\n"
                    
                    # Sayfa metnini al ve işle
                    page_text = self.extract_page_content(page)
                    text += page_text
                    
                    # Sayfa sonu işareti
                    text += "\n---\n"
                
                return text
        except Exception as e:
            st.error(f"PDF okuma hatası: {e}")
            return ""
    
    def extract_page_content(self, page) -> str:
        """Sayfa içeriğini detaylı şekilde çıkar"""
        content = []
        
        # Ana metni al
        main_text = page.extract_text()
        if main_text:
            content.append(self.clean_text(main_text))
        
        # Tabloları işle
        tables = self.extract_tables(page)
        if tables:
            content.append("\n=== TABLOLAR ===\n")
            for i, table in enumerate(tables, 1):
                content.append(f"Tablo {i}:")
                content.append(self.format_table(table))
        
        # Resimleri işle
        images = self.extract_images(page)
        if images:
            content.append("\n=== RESİMLER ===\n")
            for i, image_info in enumerate(images, 1):
                content.append(f"Resim {i}: {image_info}")
        
        # Bağlantıları işle
        links = self.extract_links(page)
        if links:
            content.append("\n=== BAĞLANTILAR ===\n")
            for link in links:
                content.append(f"Bağlantı: {link}")
        
        # Dipnotları işle
        footnotes = self.extract_footnotes(page)
        if footnotes:
            content.append("\n=== DİPNOTLAR ===\n")
            for footnote in footnotes:
                content.append(f"Dipnot: {footnote}")
        
        return "\n".join(content)
    
    def extract_tables(self, page) -> List[List[List[str]]]:
        """Sayfadaki tabloları çıkar"""
        try:
            # PyPDF2'nin tablo çıkarma özelliği sınırlı olduğu için
            # basit bir tablo algılama mantığı kullanıyoruz
            text = page.extract_text()
            lines = text.split('\n')
            tables = []
            current_table = []
            
            for line in lines:
                # Tablo satırı olabilecek desenleri kontrol et
                if self.is_table_row(line):
                    if not current_table:
                        current_table = []
                    current_table.append(self.split_table_row(line))
                elif current_table:
                    if len(current_table) > 1:  # En az 2 satırlı tablo
                        tables.append(current_table)
                    current_table = []
            
            # Son tabloyu ekle
            if current_table and len(current_table) > 1:
                tables.append(current_table)
            
            return tables
        except Exception as e:
            print(f"Tablo çıkarma hatası: {e}")
            return []
    
    def is_table_row(self, line: str) -> bool:
        """Satırın tablo satırı olup olmadığını kontrol et"""
        # Tablo satırı olabilecek desenler
        patterns = [
            r'\|\s*[^|]+\s*\|',  # | ile ayrılmış
            r'[^|]+\s*\|',       # Sağda | var
            r'\|\s*[^|]+',       # Solda | var
            r'[^|]+\s*\+',       # + ile ayrılmış
            r'\+[^+]+\+',        # + ile çevrili
            r'[^|]+\s*\|[^|]+\s*\|'  # En az 2 sütun
        ]
        
        return any(re.search(pattern, line) for pattern in patterns)
    
    def split_table_row(self, line: str) -> List[str]:
        """Tablo satırını sütunlara ayır"""
        # Önce | karakterlerine göre böl
        if '|' in line:
            cells = [cell.strip() for cell in line.split('|')]
            # Boş başlangıç ve bitiş hücrelerini kaldır
            if cells and not cells[0].strip():
                cells = cells[1:]
            if cells and not cells[-1].strip():
                cells = cells[:-1]
            return cells
        
        # + karakterlerine göre böl
        if '+' in line:
            cells = [cell.strip() for cell in line.split('+')]
            return [cell for cell in cells if cell.strip()]
        
        # Boşluklara göre böl (basit tablolar için)
        return [cell.strip() for cell in line.split() if cell.strip()]
    
    def format_table(self, table: List[List[str]]) -> str:
        """Tabloyu formatlı metne dönüştür"""
        if not table:
            return ""
        
        # Sütun genişliklerini hesapla
        col_widths = []
        for row in table:
            while len(col_widths) < len(row):
                col_widths.append(0)
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(cell))
        
        # Tabloyu formatla
        formatted_rows = []
        for row in table:
            # Eksik hücreleri doldur
            while len(row) < len(col_widths):
                row.append("")
            # Hücreleri genişliğe göre hizala
            formatted_cells = [cell.ljust(width) for cell, width in zip(row, col_widths)]
            formatted_rows.append(" | ".join(formatted_cells))
        
        return "\n".join(formatted_rows)
    
    def extract_images(self, page) -> List[str]:
        """Sayfadaki resimleri çıkar"""
        try:
            images = []
            if '/Resources' in page and '/XObject' in page['/Resources']:
                xObject = page['/Resources']['/XObject']
                for obj in xObject:
                    if xObject[obj]['/Subtype'] == '/Image':
                        images.append(f"Resim bulundu: {obj}")
            return images
        except Exception as e:
            print(f"Resim çıkarma hatası: {e}")
            return []
    
    def extract_links(self, page) -> List[str]:
        """Sayfadaki bağlantıları çıkar"""
        try:
            links = []
            if '/Annots' in page:
                for annot in page['/Annots']:
                    if annot.get_object()['/Subtype'] == '/Link':
                        if '/A' in annot.get_object():
                            link = annot.get_object()['/A']
                            if '/URI' in link:
                                links.append(link['/URI'])
            return links
        except Exception as e:
            print(f"Bağlantı çıkarma hatası: {e}")
            return []
    
    def extract_footnotes(self, page) -> List[str]:
        """Sayfadaki dipnotları çıkar"""
        try:
            text = page.extract_text()
            lines = text.split('\n')
            footnotes = []
            
            # Dipnot desenlerini kontrol et
            footnote_patterns = [
                r'^\d+\.\s',  # 1. ile başlayan
                r'^\[\d+\]\s',  # [1] ile başlayan
                r'^\(\d+\)\s',  # (1) ile başlayan
                r'^[a-z]\)\s',  # a) ile başlayan
                r'^[A-Z]\)\s'   # A) ile başlayan
            ]
            
            for line in lines:
                if any(re.match(pattern, line) for pattern in footnote_patterns):
                    footnotes.append(line.strip())
            
            return footnotes
        except Exception as e:
            print(f"Dipnot çıkarma hatası: {e}")
            return []
    
    def clean_text(self, text: str) -> str:
        """Metni temizle ve düzenle - gelişmiş temizleme"""
        if not text:
            return ""
        
        # Gereksiz boşlukları temizle
        text = ' '.join(text.split())
        
        # Özel karakterleri düzelt
        replacements = {
            '…': '...',
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
            '–': '-',
            '—': '--',
            '•': '*',
            '→': '->',
            '←': '<-',
            '↑': '^',
            '↓': 'v',
            '±': '+/-',
            '×': 'x',
            '÷': '/',
            '≠': '!=',
            '≤': '<=',
            '≥': '>=',
            '∞': 'inf',
            '°': 'derece',
            '²': '2',
            '³': '3',
            '½': '1/2',
            '¼': '1/4',
            '¾': '3/4'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Birden fazla noktalama işaretini düzelt
        text = re.sub(r'\.{2,}', '...', text)
        text = re.sub(r'\!{2,}', '!', text)
        text = re.sub(r'\?{2,}', '?', text)
        text = re.sub(r'\,{2,}', ',', text)
        text = re.sub(r'\;{2,}', ';', text)
        text = re.sub(r'\:{2,}', ':', text)
        
        # Parantezleri düzelt
        text = re.sub(r'\(\s+', '(', text)
        text = re.sub(r'\s+\)', ')', text)
        text = re.sub(r'\[\s+', '[', text)
        text = re.sub(r'\s+\]', ']', text)
        
        # Tire ve tire işaretlerini düzelt
        text = re.sub(r'\s*-\s*', '-', text)
        text = re.sub(r'\s*–\s*', '-', text)
        text = re.sub(r'\s*—\s*', '--', text)
        
        # Sayısal ifadeleri düzelt
        text = re.sub(r'(\d+)\s*\.\s*(\d+)', r'\1.\2', text)  # Ondalık sayılar
        text = re.sub(r'(\d+)\s*,\s*(\d+)', r'\1,\2', text)   # Binlik ayracı
        
        # Madde işaretlerini düzelt
        text = re.sub(r'^\s*[•\-\*]\s+', '* ', text, flags=re.MULTILINE)
        
        # Boş satırları temizle
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def create_chunks(self, text: str) -> List[str]:
        """Gelişmiş metin parçalama - akıllı bölme stratejisi"""
        # Önce temizlik
        text = self.clean_text(text)
        
        chunks = []
        current_chunk = ""
        current_length = 0
        
        # Sayfa bazlı bölme
        pages = text.split('[Sayfa')
        for page in pages:
            if not page.strip():
                continue
                
            # Sayfa numarasını ayır
            page_parts = page.split(']', 1)
            if len(page_parts) > 1:
                page_num = page_parts[0].strip()
                page_content = page_parts[1].strip()
            else:
                page_content = page.strip()
            
            # Paragraflara böl
            paragraphs = page_content.split('\n\n')
            
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                
                # Cümlelere böl
                sentences = paragraph.split('. ')
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    # Cümleyi ekle
                    test_chunk = current_chunk + " " + sentence if current_chunk else sentence
                    test_length = len(test_chunk.split())
                    
                    if test_length <= Config.CHUNK_SIZE:
                        current_chunk = test_chunk
                        current_length = test_length
                    else:
                        # Mevcut chunk'ı kaydet
                        if current_chunk and current_length > 10:
                            # Sayfa bilgisini ekle
                            if page_num:
                                current_chunk = f"[Sayfa {page_num}] {current_chunk}"
                            chunks.append(current_chunk.strip())
                        
                        # Yeni chunk başlat - overlap için son birkaç kelimeyi al
                        words = current_chunk.split()
                        if len(words) > Config.CHUNK_OVERLAP:
                            overlap_text = " ".join(words[-Config.CHUNK_OVERLAP:])
                            current_chunk = overlap_text + " " + sentence
                            current_length = len(current_chunk.split())
                        else:
                            current_chunk = sentence
                            current_length = len(sentence.split())
        
        # Son chunk'ı ekle
        if current_chunk and current_length > 10:
            chunks.append(current_chunk.strip())
        
        # Eğer çok az chunk varsa, kelime bazlı bölme de yap
        if len(chunks) < 3:
            words = text.split()
            for i in range(0, len(words), Config.CHUNK_SIZE - Config.CHUNK_OVERLAP):
                chunk = " ".join(words[i:i + Config.CHUNK_SIZE])
                if len(chunk.strip()) > 50:
                    chunks.append(chunk.strip())
        
        # Chunk'ları optimize et
        optimized_chunks = []
        for chunk in chunks:
            # Çok kısa chunk'ları birleştir
            if len(optimized_chunks) > 0 and len(chunk.split()) < 20:
                last_chunk = optimized_chunks[-1]
                if len(last_chunk.split()) + len(chunk.split()) <= Config.CHUNK_SIZE:
                    optimized_chunks[-1] = last_chunk + " " + chunk
                    continue
            
            # Çok uzun chunk'ları böl
            if len(chunk.split()) > Config.CHUNK_SIZE * 1.5:
                words = chunk.split()
                mid_point = len(words) // 2
                optimized_chunks.append(" ".join(words[:mid_point]))
                optimized_chunks.append(" ".join(words[mid_point:]))
            else:
                optimized_chunks.append(chunk)
        
        return optimized_chunks
    
    def initialize_embeddings(self):
        """Embedding modelini başlat"""
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
    
    def create_vector_store(self, chunks: List[str]) -> faiss.IndexFlatIP:
        """Vektör veritabanı oluştur"""
        self.initialize_embeddings()
        
        # Embedding'leri oluştur
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=False)
        embeddings = embeddings.astype('float32')
        
        # FAISS index oluştur
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        
        # Normalize et (cosine similarity için)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        
        return index
    
    def load_and_process_pdf(self):
        """Sabit PDF dosyasını yükle ve işle"""
        pdf_path = Config.PDF_FILE_PATH
        
        # PDF dosyası var mı kontrol et
        if not os.path.exists(pdf_path):
            st.error(f"PDF dosyası bulunamadı: {pdf_path}")
            st.info("Lütfen 'document.pdf' adlı dosyayı uygulama klasörüne ekleyin.")
            return False
        
        # PDF hash'ini kontrol et
        pdf_hash = self.get_pdf_hash(pdf_path)
        if not pdf_hash:
            return False
        
        # Cache'de var mı kontrol et
        if pdf_hash in self.cache:
            cached_data = self.cache[pdf_hash]
            if 'chunks' in cached_data and 'vector_store' in cached_data:
                self.chunks = cached_data['chunks']
                self.vector_store = cached_data['vector_store']
                self.pdf_processed = True
                return True
        
        # PDF'i işle
        with st.spinner("PDF işleniyor..."):
            text = self.extract_text_from_pdf(pdf_path)
            
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
            
            self.pdf_processed = True
            return True
    
    def semantic_search(self, query: str, k: int = 8) -> List[Dict]:
        """Gelişmiş semantic search - çoklu strateji ve genişletilmiş arama"""
        if self.vector_store is None or not self.chunks:
            return []
        
        self.initialize_embeddings()
        
        # 1. Ana semantic search - daha fazla sonuç al
        query_embedding = self.embedding_model.encode([query]).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Daha fazla sonuç al ve daha düşük eşik kullan
        scores, indices = self.vector_store.search(query_embedding, min(k * 3, len(self.chunks)))
        
        results = []
        
        # 2. Gelişmiş scoring ve filtreleme
        for i, score in zip(indices[0], scores[0]):
            if i >= 0 and score > 0.1:  # Eşiği daha da düşür
                chunk_text = self.chunks[i]
                
                # 3. Gelişmiş keyword matching
                query_words = set(query.lower().split())
                chunk_words = set(chunk_text.lower().split())
                
                # Türkçe karakterleri normalize et
                query_normalized = self.normalize_turkish(query.lower())
                chunk_normalized = self.normalize_turkish(chunk_text.lower())
                
                # Kelime eşleşmelerini hesapla
                exact_matches = len(query_words.intersection(chunk_words))
                partial_matches = sum(1 for qw in query_words if any(cw.startswith(qw) or qw.startswith(cw) for cw in chunk_words))
                
                # 4. Cümle benzerliği
                sentence_similarity = 0
                query_sentences = query_normalized.split('.')
                chunk_sentences = chunk_normalized.split('.')
                
                for qs in query_sentences:
                    for cs in chunk_sentences:
                        if len(qs) > 3 and len(cs) > 3:
                            if qs in cs or cs in qs:
                                sentence_similarity += 1
                
                # 5. N-gram benzerliği
                ngram_similarity = 0
                for n in range(2, 4):  # 2-gram ve 3-gram
                    query_ngrams = set(' '.join(query_normalized.split()[i:i+n]) for i in range(len(query_normalized.split())-n+1))
                    chunk_ngrams = set(' '.join(chunk_normalized.split()[i:i+n]) for i in range(len(chunk_normalized.split())-n+1))
                    ngram_similarity += len(query_ngrams.intersection(chunk_ngrams)) / max(len(query_ngrams), 1)
                
                # 6. Final score hesapla - daha dengeli ağırlıklar
                final_score = (
                    score * 0.4 +  # Semantic similarity
                    (exact_matches / max(len(query_words), 1)) * 0.2 +  # Exact matches
                    (partial_matches / max(len(query_words), 1)) * 0.2 +  # Partial matches
                    (sentence_similarity / max(len(query_sentences), 1)) * 0.1 +  # Sentence similarity
                    ngram_similarity * 0.1  # N-gram similarity
                )
                
                results.append({
                    'text': chunk_text,
                    'score': final_score,
                    'semantic_score': score,
                    'exact_matches': exact_matches,
                    'partial_matches': partial_matches,
                    'sentence_similarity': sentence_similarity,
                    'ngram_similarity': ngram_similarity
                })
        
        # 7. Score'a göre sırala ve en iyileri al
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # En az 0.15 score'u olanları al, ama minimum 3 tane
        filtered_results = [r for r in results if r['score'] > 0.15]
        if len(filtered_results) < 3 and results:
            filtered_results = results[:3]
        
        return filtered_results[:k]
    
    def normalize_turkish(self, text: str) -> str:
        """Türkçe karakterleri normalize et"""
        replacements = {
            'ç': 'c', 'ğ': 'g', 'ı': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u',
            'Ç': 'C', 'Ğ': 'G', 'İ': 'I', 'Ö': 'O', 'Ş': 'S', 'Ü': 'U'
        }
        for tr_char, en_char in replacements.items():
            text = text.replace(tr_char, en_char)
        return text
    
    def generate_response(self, question: str, context_results: List[Dict]) -> str:
        """Gelişmiş cevap üretimi - akıllı ve doğal yanıt sistemi"""
        # Basit sorular için özel kontrol
        simple_questions = {
            "merhaba": "Merhaba! Ben size yardımcı olmak için buradayım. Ne öğrenmek istersiniz?",
            "selam": "Selam! Size nasıl yardımcı olabilirim?",
            "nasılsın": "İyiyim, teşekkür ederim! Sizinle sohbet etmek ve bilgi paylaşmak için hazırım.",
            "teşekkür": "Rica ederim! Başka sorularınız varsa yardımcı olmaktan mutluluk duyarım.",
            "yardım": "Size şu konularda yardımcı olabilirim:\n* Detaylı bilgi analizi\n* Tablo ve listeler\n* Görsel açıklamaları\n* Başlık ve alt başlıklar\n* Karmaşık konuların açıklamaları\n\nNe hakkında bilgi almak istersiniz?",
            "görüşürüz": "Görüşmek üzere! İyi günler dilerim.",
            "hoşça kal": "Hoşça kalın! Tekrar görüşmek üzere.",
            "sağol": "Rica ederim! Başka sorularınız olursa yardımcı olmaktan mutluluk duyarım."
        }
        
        # Basit selamlaşma ve teşekkür kalıpları
        greeting_patterns = [
            r'^merhaba.*$',
            r'^selam.*$',
            r'^nasılsın.*$',
            r'^teşekkür.*$',
            r'^sağol.*$',
            r'^görüşürüz.*$',
            r'^hoşça kal.*$',
            r'^yardım.*$'
        ]
        
        # Soruyu normalize et ve anlamaya çalış
        question_lower = question.lower().strip()
        question_analysis = self.analyze_question_detailed(question)
        
        # Soruyu düzelt ve anlamaya çalış
        corrected_question = self.correct_and_understand_question(question, question_analysis)
        
        # Basit soru kontrolü
        for pattern in greeting_patterns:
            if re.match(pattern, question_lower):
                for key, response in simple_questions.items():
                    if key in question_lower:
                        return response
                return "Merhaba! Size nasıl yardımcı olabilirim?"
        
        # Eğer context yoksa, soruyu anlamaya çalış ve uygun bir yanıt ver
        if not context_results:
            return self.generate_understanding_response(corrected_question, question_analysis)
        
        # Soru karmaşıklığına göre prompt seç
        if question_analysis['complexity'] == 'low':
            prompt = self.generate_simple_prompt(corrected_question, context_results)
        else:
            prompt = self.generate_detailed_prompt(corrected_question, context_results, question_analysis)
        
        try:
            response = self.model.generate_content(prompt)
            generated_response = response.text.strip()
            
            # Cevabı karmaşıklığa göre formatla
            if question_analysis['complexity'] == 'low':
                formatted_response = self.format_simple_response(generated_response)
            else:
                formatted_response = self.format_response_detailed(
                    generated_response, 
                    question_analysis,
                    self.analyze_context_detailed(context_results)
                )
            
            # Debug bilgisini ekle
            if st.session_state.get('debug_mode', False):
                formatted_response += f"\n\n[Debug - Analiz: {question_analysis['type']}, Güven: {question_analysis['confidence']:.2f}]"
            
            return formatted_response
            
        except Exception as e:
            return f"Üzgünüm, bu soruyu yanıtlarken bir hata oluştu. Sorunuzu farklı bir şekilde ifade edebilir misiniz?"

    def correct_and_understand_question(self, question: str, analysis: Dict) -> str:
        """Soruyu düzelt ve anlamaya çalış - gelişmiş yazım düzeltme"""
        # Boşluk düzeltmeleri
        question = re.sub(r'\s+', ' ', question)  # Fazla boşlukları temizle
        question = re.sub(r'\s+([.,!?])', r'\1', question)  # Noktalama öncesi boşluk
        question = re.sub(r'([.,!?])([^\s])', r'\1 \2', question)  # Noktalama sonrası boşluk
        
        # Türkçe karakter düzeltmeleri
        turkish_corrections = {
            'i': 'i', 'I': 'İ',  # Türkçe i düzeltmesi
            's': 'ş', 'S': 'Ş',  # ş düzeltmesi
            'c': 'ç', 'C': 'Ç',  # ç düzeltmesi
            'g': 'ğ', 'G': 'Ğ',  # ğ düzeltmesi
            'o': 'ö', 'O': 'Ö',  # ö düzeltmesi
            'u': 'ü', 'U': 'Ü'   # ü düzeltmesi
        }
        
        # Yaygın yazım hataları
        common_typos = {
            'belgede': '',
            'pdfte': '',
            'pdf\'te': '',
            'belgedeki': '',
            'belgenin': '',
            'belgeye göre': '',
            'belgeden': '',
            'nedir': 'nedir',
            'nasıl': 'nasıl',
            'ne zaman': 'ne zaman',
            'nerede': 'nerede',
            'kim': 'kim',
            'hangi': 'hangi',
            'kaç': 'kaç',
            'neden': 'neden',
            'niye': 'niçin',
            'niçin': 'neden',
            'açıklama': 'açıklama',
            'detay': 'detay',
            'bilgi': 'bilgi',
            'anlat': 'anlat',
            'söyle': 'söyle',
            'göster': 'göster',
            'ver': 'ver',
            'bul': 'bul',
            'ara': 'ara',
            'sor': 'sor'
        }
        
        # Kelime bazlı düzeltmeler
        words = question.split()
        corrected_words = []
        
        for word in words:
            # Türkçe karakter düzeltmesi
            for wrong, correct in turkish_corrections.items():
                if wrong in word:
                    # Kelimenin başındaki ve sonundaki karakterleri koru
                    if word.startswith(wrong):
                        word = correct + word[1:]
                    if word.endswith(wrong):
                        word = word[:-1] + correct
            
            # Yaygın yazım hatalarını düzelt
            word_lower = word.lower()
            if word_lower in common_typos:
                corrected_word = common_typos[word_lower]
                # Orijinal kelimenin büyük/küçük harf durumunu koru
                if word[0].isupper():
                    corrected_word = corrected_word[0].upper() + corrected_word[1:]
                word = corrected_word
            
            corrected_words.append(word)
        
        # Düzeltilmiş kelimeleri birleştir
        corrected_question = ' '.join(corrected_words)
        
        # Noktalama düzeltmeleri
        punctuation_corrections = {
            ',,': ',',
            '..': '.',
            '!!': '!',
            '??': '?',
            '.,': '.',
            '.,.': '.',
            '!?': '?',
            '?!': '?',
            '...': '...',
            '....': '...'
        }
        
        for wrong, correct in punctuation_corrections.items():
            corrected_question = corrected_question.replace(wrong, correct)
        
        # Soru işareti kontrolü
        if not corrected_question.endswith('?') and analysis['type'] != 'greeting':
            # Soru kelimeleri varsa soru işareti ekle
            question_words = ['nedir', 'nasıl', 'ne zaman', 'nerede', 'kim', 'hangi', 'kaç', 'neden', 'niçin']
            if any(word in corrected_question.lower() for word in question_words):
                corrected_question += '?'
        
        return corrected_question.strip()

    def generate_understanding_response(self, question: str, analysis: Dict) -> str:
        """Soruyu anlamaya çalışarak uygun yanıt üret - gelişmiş yanıt sistemi"""
        # Dil kalitesi düzeltmesi gerekiyorsa
        if analysis['language_quality'] != 'good':
            if analysis['language_quality'] == 'needs_spacing_correction':
                return "Sorunuzu daha anlaşılır hale getirdim. Şimdi size yardımcı olabilirim."
            elif analysis['language_quality'] == 'needs_punctuation_correction':
                return "Sorunuzu daha net anladım. Size yardımcı olmaya çalışacağım."
            elif analysis['language_quality'] == 'needs_character_correction':
                return "Sorunuzu anlamaya çalıştım. Size nasıl yardımcı olabilirim?"
        
        # Açıklama ihtiyacı varsa
        if analysis['needs_clarification']:
            if not analysis['is_question']:
                return "Sizi daha iyi anlayabilmem için biraz daha detay verebilir misiniz?"
            elif analysis['confidence'] < 0.3:
                return "Bu konu hakkında bilgim yok. Başka bir konuda size yardımcı olabilirim."
        
        # Soru tipine göre özel yanıtlar
        if analysis['type'] == 'greeting':
            return "Merhaba! Size nasıl yardımcı olabilirim?"
        elif analysis['type'] == 'definition':
            return "Bu kavram hakkında bilgim yok. Başka bir konuda size yardımcı olabilirim."
        elif analysis['type'] == 'numeric':
            return "Bu sayısal veri hakkında bilgim yok. Başka bir konuda size yardımcı olabilirim."
        elif analysis['type'] == 'comparison':
            return "Bu karşılaştırma hakkında bilgim yok. Başka bir konuda size yardımcı olabilirim."
        
        return "Bu konu hakkında bilgim yok. Başka bir konuda size yardımcı olabilirim."

    def generate_simple_prompt(self, question: str, context_results: List[Dict]) -> str:
        """Basit sorular için optimize edilmiş prompt"""
        return f"""
        Sen, verilen içerik üzerine eğitilmiş, sadece bu içeriğe odaklanan bir yapay zeka asistanısın.

        KURALLAR:
        1. SADECE verilen içerikteki bilgileri kullan
        2. İçerik dışında bilgi verme
        3. Doğal ve akıcı bir dil kullan
        4. Kısa ve öz cevaplar ver (en fazla 2-3 cümle)
        5. Gereksiz detaylardan kaçın
        6. Bilgiyi ezberlenmiş gibi değil, anlamış gibi kullan
        7. "Belgede", "PDF'te" gibi ifadeler kullanma
        8. Bilgiyi doğrudan ve net şekilde sun
        9. Kullanıcıya yardımcı olmaya odaklan
        10. İçerik dışı sorulara nazikçe yönlendir
        11. Birincil ağızdan konuş ("Ben" diyerek)
        12. Bilmediğin konularda "Bu konu hakkında bilgim yok" de
        13. Asla "Bu bilgi metinde yok" veya "Bu bilgi PDF'te yok" deme

        ÖNEMLİ: 
        - Sadece verilen içerikteki bilgileri kullan
        - İçerik dışında bilgi verme
        - Bilmediğin konularda "Bu konu hakkında bilgim yok" de
        - Birincil ağızdan konuş ("Ben" diyerek)
        - Asla "Bu bilgi metinde yok" veya "Bu bilgi PDF'te yok" deme

        İÇERİK:
        {self.format_context_for_simple_prompt(context_results)}
        
        SORU: {question}
        
        YANIT (birincil ağızdan, doğal ve akıcı):
        """
    
    def generate_detailed_prompt(self, question: str, context_results: List[Dict], question_analysis: Dict) -> str:
        """Karmaşık sorular için detaylı prompt"""
        return f"""
        Sen, verilen içerik üzerine eğitilmiş, sadece bu içeriğe odaklanan bir yapay zeka asistanısın.

        ## Analiz Bilgileri:
        - Soru Tipi: {question_analysis['type']}
        - Soru Kategorisi: {question_analysis['category']}
        - Beklenen Format: {question_analysis['expected_format']}
        - Güven Skoru: {question_analysis['confidence']:.2f}
        - Anahtar Kelimeler: {', '.join(question_analysis['keywords'])}

        ## İLKELER:
        1. 🎯 **İÇERİK ODAKLI YANITLAR**:
           - SADECE verilen içerikteki bilgileri kullan
           - İçerik dışında bilgi verme
           - İçerik dışı sorulara nazikçe yönlendir
           - Bilmediğin konularda "Bu konu hakkında bilgim yok" de
           - Asla "Bu bilgi metinde yok" veya "Bu bilgi PDF'te yok" deme

        2. 🧠 **Anlamaya Çalış**:
           - Soruyu düzelt, amacını anla
           - Kullanıcının niyetini kavra
           - Mantıklı yorumla
           - İçerik sınırları içinde kal
           - Birincil ağızdan konuş ("Ben" diyerek)

        3. 🧾 **Yapılandırılmış Yanıtlar**:
           - Karmaşık konular için maddeler ve tablolar kullan
           - Basit sorulara kısa ama mantıklı cevaplar ver
           - Gerektiğinde örnek ver ve açıkla
           - Sadece içerikteki örnekleri kullan
           - Birincil ağızdan konuş ("Ben" diyerek)

        4. 🔍 **Kusursuz Anlama**:
           - Tüm veri türlerini anla
           - İlişkili bilgileri birleştir
           - Bütünsel şekilde sun
           - İçerik sınırları içinde kal
           - Bilmediğin konularda "Bu konu hakkında bilgim yok" de

        5. ⚡ **Hızlı ve Akıcı**:
           - Net ve mantıklı cevaplar ver
           - Gereksiz detaylardan kaçın
           - Özü koru
           - İçerik odaklı ol
           - Birincil ağızdan konuş ("Ben" diyerek)

        6. 🧠 **Zekâ Gibi Davran**:
           - Bilgiyi anlamış gibi kullan
           - Özgün ve yaratıcı ol
           - Doğal konuş ama net ol
           - İçerik sınırları içinde kal
           - Bilmediğin konularda "Bu konu hakkında bilgim yok" de

        7. 🛑 **Yapma**:
           - İçerik dışı bilgi verme
           - "Bu bilgi yok" gibi cevaplar verme
           - "Daha net sor" gibi cevaplar verme
           - Belge referansları kullanma
           - "Bu bilgi metinde yok" veya "Bu bilgi PDF'te yok" deme

        8. 🧩 **Bağlantı Kur**:
           - Önceki soruları hatırla
           - Mantıklı devamlar üret
           - Bağlam kur
           - İçerik sınırları içinde kal
           - Birincil ağızdan konuş ("Ben" diyerek)

        9. 🤖 **Kimliğin**:
           - İçerik odaklı asistan
           - Bilgi analisti
           - Cevap üretici
           - Birincil ağızdan konuşan asistan

        ## İçerik:
        {self.format_context_for_detailed_prompt(self.analyze_context_detailed(context_results))}

        ## Soru:
        {question}

        ## Yanıt (birincil ağızdan, doğal ve akıcı):
        """
    
    def format_context_for_simple_prompt(self, context_results: List[Dict]) -> str:
        """Basit prompt için context formatla"""
        formatted = []
        
        for result in context_results[:2]:  # Sadece en alakalı 2 sonucu kullan
            text = result['text']
            score = result['score']
            
            # İçerik dışı bilgileri temizle
            text = re.sub(r'\[Sayfa \d+\]', '', text)  # Sayfa numaralarını kaldır
            text = re.sub(r'=== PDF BİLGİLERİ ===.*?===', '', text, flags=re.DOTALL)  # PDF bilgilerini kaldır
            
            formatted.append(f"[İlgililik: %{score*100:.1f}]\n{text}")
        
        return "\n".join(formatted)
    
    def format_simple_response(self, response: str) -> str:
        """Basit yanıtı formatla"""
        if not response:
            return response
        
        # Gereksiz boşlukları temizle
        response = ' '.join(response.split())
        
        # Çok uzun cümleleri böl
        sentences = response.split('. ')
        if len(sentences) > 3:
            response = '. '.join(sentences[:3]) + '.'
        
        # Markdown formatlamasını basitleştir
        response = re.sub(r'\*\*(.*?)\*\*', r'\1', response)  # Kalın
        response = re.sub(r'\*(.*?)\*', r'\1', response)      # İtalik
        
        return response.strip()
    
    def analyze_context_detailed(self, context_results: List[Dict]) -> Dict:
        """Context'leri detaylı analiz et"""
        analyzed = {
            'content_types': {
                'tables': [],
                'lists': [],
                'text': [],
                'numbers': [],
                'comparisons': [],
                'definitions': [],
                'images': [],
                'formatted_text': [],
                'metadata': []
            },
            'semantic': {
                'keywords': set(),
                'topics': set(),
                'entities': set(),
                'relationships': []
            }
        }
        
        for result in context_results:
            text = result['text']
            score = result['score']
            
            # İçerik tipi analizi
            if '|' in text or '+' in text or self.is_table_row(text):
                analyzed['content_types']['tables'].append({
                    'content': text,
                    'score': score,
                    'format': self.detect_table_format(text)
                })
            
            elif any(text.startswith(marker) for marker in ['*', '-', '•', '1.', 'a)', 'A)']):
                analyzed['content_types']['lists'].append({
                    'content': text,
                    'score': score,
                    'type': self.detect_list_type(text)
                })
            
            elif re.search(r'\d+[\.,]?\d*', text):
                analyzed['content_types']['numbers'].append({
                    'content': text,
                    'score': score,
                    'numbers': self.extract_numbers(text)
                })
            
            # Format analizi
            if re.search(r'\*\*.*\*\*', text):  # Kalın
                analyzed['content_types']['formatted_text'].append({
                    'content': text,
                    'format': 'bold',
                    'score': score
                })
            elif re.search(r'\*.*\*', text):  # İtalik
                analyzed['content_types']['formatted_text'].append({
                    'content': text,
                    'format': 'italic',
                    'score': score
                })
            
            # Anahtar kelime ve konu analizi
            words = text.lower().split()
            analyzed['semantic']['keywords'].update(w for w in words if len(w) > 3)
            
            # Varlık analizi (isim, tarih, sayı vb.)
            analyzed['semantic']['entities'].update(self.extract_entities(text))
        
        return analyzed
    
    def analyze_question_detailed(self, question: str) -> Dict:
        """Soruyu detaylı analiz et - gelişmiş analiz"""
        question = question.lower()
        
        # Soru tipi analizi - genişletilmiş
        question_types = {
            'greeting': ['merhaba', 'selam', 'nasılsın', 'teşekkür', 'sağol', 'görüşürüz', 'hoşça kal'],
            'table': ['tablo', 'liste', 'sırala', 'düzenle', 'karşılaştır', 'tabloda', 'listede'],
            'list': ['say', 'listele', 'madde', 'öğe', 'unsur', 'sırala', 'numara'],
            'comparison': ['fark', 'benzer', 'karşılaştır', 'vs', 'versus', 'aynı', 'farklı'],
            'numeric': ['kaç', 'sayı', 'oran', 'yüzde', 'miktar', 'adet', 'tane'],
            'definition': ['nedir', 'ne demek', 'tanım', 'anlam', 'açıkla', 'ifade'],
            'explanation': ['nasıl', 'neden', 'açıkla', 'anlat', 'göster', 'belirt'],
            'summary': ['özet', 'özetle', 'kısaca', 'genel', 'tüm', 'hepsi'],
            'search': ['ara', 'bul', 'göster', 'ver', 'söyle', 'hangi', 'nerede']
        }
        
        # Soru kategorisi analizi - genişletilmiş
        categories = {
            'technical': ['teknik', 'sistem', 'yapı', 'mekanizma', 'işleyiş', 'çalışma'],
            'conceptual': ['kavram', 'fikir', 'düşünce', 'teori', 'prensip', 'ilke'],
            'practical': ['uygulama', 'kullanım', 'pratik', 'örnek', 'nasıl', 'yapılır'],
            'analytical': ['analiz', 'değerlendirme', 'inceleme', 'karşılaştırma', 'fark'],
            'general': []
        }
        
        # Format beklentisi analizi - genişletilmiş
        format_expectations = {
            'table': ['tablo', 'liste', 'sıralı', 'düzenli', 'karşılaştırmalı'],
            'list': ['madde', 'öğe', 'sıralı', 'numaralı', 'başlıklı'],
            'comparison': ['karşılaştırma', 'yan yana', 'tablo', 'liste'],
            'numeric': ['sayısal', 'grafik', 'tablo', 'liste', 'oran'],
            'text': ['açıklama', 'anlatım', 'paragraf', 'detay', 'bilgi']
        }
        
        # Analiz sonuçları
        analysis = {
            'type': 'general',
            'category': 'general',
            'expected_format': 'text',
            'confidence': 0.0,
            'keywords': set(),
            'complexity': 'medium',
            'is_question': False,
            'needs_clarification': False,
            'language_quality': 'good'
        }
        
        # Soru olup olmadığını kontrol et
        question_indicators = ['?', 'nedir', 'nasıl', 'ne zaman', 'nerede', 'kim', 'hangi', 'kaç', 'neden', 'niçin']
        analysis['is_question'] = any(indicator in question for indicator in question_indicators)
        
        # Dil kalitesini kontrol et
        if re.search(r'\s{2,}', question):  # Fazla boşluk
            analysis['language_quality'] = 'needs_spacing_correction'
        if re.search(r'[.,!?]{2,}', question):  # Fazla noktalama
            analysis['language_quality'] = 'needs_punctuation_correction'
        if re.search(r'[^a-zA-ZğüşıöçĞÜŞİÖÇ\s.,!?]', question):  # Geçersiz karakterler
            analysis['language_quality'] = 'needs_character_correction'
        
        # Soru tipini belirle
        for qtype, keywords in question_types.items():
            if any(keyword in question for keyword in keywords):
                analysis['type'] = qtype
                analysis['confidence'] += 0.3
                break
        
        # Kategoriyi belirle
        for category, keywords in categories.items():
            if any(keyword in question for keyword in keywords):
                analysis['category'] = category
                analysis['confidence'] += 0.2
                break
        
        # Format beklentisini belirle
        for format_type, keywords in format_expectations.items():
            if any(keyword in question for keyword in keywords):
                analysis['expected_format'] = format_type
                analysis['confidence'] += 0.2
                break
        
        # Anahtar kelimeleri çıkar
        words = question.split()
        analysis['keywords'] = {w for w in words if len(w) > 2}  # 2 karakterden uzun kelimeler
        
        # Karmaşıklık analizi
        word_count = len(words)
        if word_count > 15:
            analysis['complexity'] = 'high'
        elif word_count < 5:
            analysis['complexity'] = 'low'
        
        # Açıklama ihtiyacı kontrolü
        if analysis['confidence'] < 0.3 or (not analysis['is_question'] and analysis['type'] == 'general'):
            analysis['needs_clarification'] = True
        
        # Güven skorunu normalize et
        analysis['confidence'] = min(analysis['confidence'], 1.0)
        
        return analysis
    
    def format_context_for_detailed_prompt(self, analyzed_context: Dict) -> str:
        """Context'i detaylı prompt için formatla"""
        formatted = []
        
        # İçerik tiplerini formatla
        for content_type, items in analyzed_context['content_types'].items():
            if items:
                formatted.append(f"\n### {content_type.upper()} ###")
                for item in items:
                    if isinstance(item, dict):
                        content = item.get('content', '')
                        score = item.get('score', 0)
                        format_info = item.get('format', '')
                        
                        # İçerik dışı bilgileri temizle
                        content = re.sub(r'\[Sayfa \d+\]', '', content)
                        content = re.sub(r'=== PDF BİLGİLERİ ===.*?===', '', content, flags=re.DOTALL)
                        
                        # Format bilgisini ekle
                        format_str = f" [{format_info}]" if format_info else ""
                        
                        formatted.append(f"[İlgililik: %{score*100:.1f}]{format_str}\n{content}")
                    else:
                        formatted.append(str(item))
        
        # Semantik bilgileri ekle
        if analyzed_context['semantic']['keywords']:
            formatted.append(f"\n### ANAHTAR KELİMELER ###")
            formatted.append(", ".join(sorted(analyzed_context['semantic']['keywords'])))
        
        return "\n".join(formatted)
    
    def format_response_detailed(self, response: str, question_analysis: Dict, analyzed_context: Dict) -> str:
        """Cevabı detaylı şekilde formatla"""
        if not response:
            return response
        
        # Markdown formatlaması
        response = self.format_markdown(response)
        
        # Soru tipine göre özel formatlama
        if question_analysis['type'] == 'table':
            response = self.format_table_response(response)
        elif question_analysis['type'] == 'list':
            response = self.format_list_response(response)
        elif question_analysis['type'] == 'comparison':
            response = self.format_comparison_response(response)
        elif question_analysis['type'] == 'numeric':
            response = self.format_numeric_response(response)
        
        return response.strip()
    
    def format_markdown(self, text: str) -> str:
        """Markdown formatlamasını düzelt"""
        # Başlıkları düzelt
        text = re.sub(r'^#+\s+', lambda m: '#' * min(len(m.group()), 3) + ' ', text, flags=re.MULTILINE)
        
        # Listeleri düzelt
        text = re.sub(r'^\s*[-•]\s+', '* ', text, flags=re.MULTILINE)
        
        # Tabloları düzelt
        text = re.sub(r'\|\s*\|', '|', text)
        text = re.sub(r'\|\s*\n\s*\|', '|\n|', text)
        
        # Vurguları düzelt
        text = re.sub(r'\*\*(.*?)\*\*', r'**\1**', text)  # Kalın
        text = re.sub(r'\*(.*?)\*', r'*\1*', text)        # İtalik
        
        return text
    
    def format_table_response(self, text: str) -> str:
        """Tablo yanıtını formatla"""
        lines = text.split('\n')
        formatted_lines = []
        
        for i, line in enumerate(lines):
            if '|' in line:
                # Başlık satırından sonra çizgi ekle
                if i > 0 and '|' not in lines[i-1]:
                    cells = line.split('|')
                    separator = '|' + '|'.join(['---' for _ in cells[1:-1]]) + '|'
                    formatted_lines.append(line)
                    formatted_lines.append(separator)
                else:
                    formatted_lines.append(line)
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def format_list_response(self, text: str) -> str:
        """Liste yanıtını formatla"""
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            # Liste öğesi kontrolü
            if re.match(r'^\s*[\d]+[\.\)]\s+', line):  # Numaralı liste
                formatted_lines.append(line.strip())
            elif re.match(r'^\s*[a-zA-Z][\.\)]\s+', line):  # Alfabetik liste
                formatted_lines.append(line.strip())
            elif re.match(r'^\s*[-•*]\s+', line):  # Madde işaretli liste
                formatted_lines.append('* ' + line.strip().lstrip('-•* '))
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def format_comparison_response(self, text: str) -> str:
        """Karşılaştırma yanıtını formatla"""
        if '|' not in text:
            # Karşılaştırma tablosu oluştur
            lines = text.split('\n')
            if len(lines) >= 2:
                # İki sütunlu tablo oluştur
                table = "| Özellik | Açıklama |\n|---------|----------|\n"
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        table += f"| {key.strip()} | {value.strip()} |\n"
                return table
        return text
    
    def format_numeric_response(self, text: str) -> str:
        """Sayısal yanıtı formatla"""
        # Sayıları binlik ayracı ile formatla
        text = re.sub(r'(\d)(?=(\d{3})+(?!\d))', r'\1.', text)
        
        # Yüzde işaretlerini düzelt
        text = re.sub(r'(\d+)\s*%', r'%\1', text)
        
        # Ondalık sayıları düzelt
        text = re.sub(r'(\d+)\s*,\s*(\d+)', r'\1,\2', text)
        
        return text
    
    def detect_table_format(self, text: str) -> str:
        """Tablo formatını tespit et"""
        if '|' in text:
            return 'markdown'
        elif '+' in text:
            return 'ascii'
        else:
            return 'simple'
    
    def detect_list_type(self, text: str) -> str:
        """Liste tipini tespit et"""
        if re.match(r'^\d+\.', text):
            return 'numbered'
        elif re.match(r'^[a-zA-Z]\.', text):
            return 'alphabetic'
        else:
            return 'bulleted'
    
    def extract_numbers(self, text: str) -> List[float]:
        """Metinden sayıları çıkar"""
        numbers = []
        for match in re.finditer(r'\d+[\.,]?\d*', text):
            try:
                num = float(match.group().replace(',', '.'))
                numbers.append(num)
            except ValueError:
                continue
        return numbers
    
    def extract_entities(self, text: str) -> Set[str]:
        """Metinden varlıkları çıkar (isim, tarih, sayı vb.)"""
        entities = set()
        
        # Tarih formatları
        date_patterns = [
            r'\d{1,2}\.\d{1,2}\.\d{2,4}',  # GG.AA.YYYY
            r'\d{1,2}/\d{1,2}/\d{2,4}',    # GG/AA/YYYY
            r'\d{1,2}-\d{1,2}-\d{2,4}'     # GG-AA-YYYY
        ]
        
        for pattern in date_patterns:
            for match in re.finditer(pattern, text):
                entities.add(match.group())
        
        # Büyük harfle başlayan kelimeler (olası isimler)
        for word in text.split():
            if word[0].isupper() and len(word) > 2:
                entities.add(word)
        
        return entities
    
    def chat(self, question: str) -> str:
        """Gelişmiş chat fonksiyonu"""
        if not self.pdf_processed or not self.vector_store or not self.chunks:
            return "PDF henüz işlenmedi. Lütfen uygulamayı yeniden başlatın."
        
        # Semantic search - daha fazla sonuç al
        relevant_results = self.semantic_search(question, k=8)
        
        if not relevant_results:
            # Eğer hiç sonuç yoksa, basit kelime araması yap
            fallback_results = self.fallback_search(question)
            if fallback_results:
                return self.generate_response(question, fallback_results)
            else:
                return "Bu konuyla ilgili belgede bilgi bulunamadı. Farklı kelimeler kullanarak tekrar deneyin."
        
        # Cevap üret
        response = self.generate_response(question, relevant_results)
        return response
    
    def fallback_search(self, query: str) -> List[Dict]:
        """Basit kelime bazlı arama (fallback)"""
        query_words = set(query.lower().split())
        results = []
        
        for i, chunk in enumerate(self.chunks):
            chunk_lower = chunk.lower()
            
            # Kelime eşleşmesi say
            matches = 0
            for word in query_words:
                if len(word) > 2 and word in chunk_lower:
                    matches += 1
            
            if matches > 0:
                score = matches / len(query_words)
                results.append({
                    'text': chunk,
                    'score': score,
                    'semantic_score': 0,
                    'keyword_score': score,
                    'substring_score': matches
                })
        
        # Score'a göre sırala
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:5]

def main():
    st.set_page_config(
        page_title="PDF Chatbot",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 PDF Chatbot")
    st.markdown("Belgenizdeki içerik hakkında sorular sorun!")
    
    # Chatbot'u session state'de sakla
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = PDFChatbot()
        
    # PDF'i otomatik yükle (sadece bir kez)
    if 'pdf_loaded' not in st.session_state:
        with st.spinner("PDF yükleniyor..."):
            success = st.session_state.chatbot.load_and_process_pdf()
            if success:
                st.session_state.pdf_loaded = True
                st.success(f"✅ PDF başarıyla yüklendi! {len(st.session_state.chatbot.chunks)} parça işlendi.")
            else:
                st.session_state.pdf_loaded = False
    
    # Ana alan - Chat interface
    col1, col2 = st.columns([3, 1])
    
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
        if st.session_state.get('pdf_loaded', False):
            question = st.chat_input("Belgeniz hakkında bir soru sorun...")
            
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
            st.error("PDF yüklenemedi. Lütfen 'document.pdf' dosyasının uygulama klasöründe olduğundan emin olun.")
    
    with col2:
        st.header("ℹ️ Durum")
        
        if st.session_state.get('pdf_loaded', False):
            st.success("✅ PDF yüklendi")
            st.info(f"📊 {len(st.session_state.chatbot.chunks)} metin parçası")
            st.info(f"📄 Dosya: {Config.PDF_FILE_PATH}")
        else:
            st.error("❌ PDF yüklenemedi")
            st.warning(f"⚠️ '{Config.PDF_FILE_PATH}' dosyası bulunamadı")
        
        st.markdown("---")
        st.markdown("**💡 Kullanım İpuçları:**")
        st.markdown("• Açık ve spesifik sorular sorun")
        st.markdown("• PDF'inizin dilinde soru sorun")
        st.markdown("• Belgedeki anahtar kelimeleri kullanın")
        
        # Debug modu toggle
        if st.checkbox("🔍 Debug Modu", help="Arama detaylarını göster"):
            st.session_state.debug_mode = True
        else:
            st.session_state.debug_mode = False
        
        # Cache temizleme
        if st.button("🗑️ Cache Temizle"):
            if os.path.exists(st.session_state.chatbot.cache_file):
                os.remove(st.session_state.chatbot.cache_file)
            st.session_state.chatbot.cache = {}
            st.success("Cache temizlendi!")
        
        # PDF yeniden yükleme
        if st.button("🔄 PDF'i Yeniden Yükle"):
            if 'pdf_loaded' in st.session_state:
                del st.session_state['pdf_loaded']
            st.rerun()

if __name__ == "__main__":
    main()
