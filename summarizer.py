#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

class NewsSummarizer:
    def __init__(self):
        self.model_name = "facebook/bart-large-cnn"
        self.tokenizer = None
        self.model = None
        self.summarizer = None
        self._load_model()
    
    def _load_model(self):
        """Загрузка предобученной модели для суммаризации"""
        try:
            print(f"Loading model {self.model_name}...", file=sys.stderr)
            
            # Используем pipeline для удобства
            self.summarizer = pipeline(
                "summarization",
                model=self.model_name,
                tokenizer=self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device=0 if torch.cuda.is_available() else -1
            )
            
            print("Model loaded successfully", file=sys.stderr)
            
        except Exception as e:
            print(f"Error loading model: {e}", file=sys.stderr)
            raise
    
    def preprocess_text(self, text, title=""):
        """Предобработка текста - объединение заголовка и текста"""
        if title and not title.lower().endswith(('.', '!', '?')):
            title += '.'
        
        # Объединяем заголовок и текст, если заголовок информативный
        if title and len(title.split()) > 2:
            full_text = f"{title} {text}"
        else:
            full_text = text
        
        return full_text
    
    def summarize(self, text, title=""):
        """Генерация краткого саммари"""
        try:
            # Предобработка текста
            processed_text = self.preprocess_text(text, title)
            
            # Генерация саммари с ограничением длины
            summary = self.summarizer(
                processed_text,
                max_length=150,
                min_length=50,
                do_sample=False,
                length_penalty=2.0,
                num_beams=4
            )
            
            result = summary[0]['summary_text']
            
            # Постобработка - убедимся, что это 2-3 предложения
            sentences = result.split('. ')
            if len(sentences) > 3:
                result = '. '.join(sentences[:3]) + '.'
            
            return result.strip()
            
        except Exception as e:
            print(f"Error during summarization: {e}", file=sys.stderr)
            return "Summary generation failed."

def main():
    # Инициализация суммаризатора
    summarizer = NewsSummarizer()
    
    try:
        # Чтение входных данных из stdin
        input_data = sys.stdin.read()
        request_data = json.loads(input_data)
        
        text = request_data.get("text", "")
        title = request_data.get("title", "")
        
        if not text:
            raise ValueError("No text provided for summarization")
        
        # Генерация саммари
        summary = summarizer.summarize(text, title)
        
        # Формирование ответа
        response = {
            "summary": summary,
            "model": summarizer.model_name
        }
        
        # Вывод результата в stdout
        print(json.dumps(response, ensure_ascii=False))
        
    except json.JSONDecodeError as e:
        error_response = {
            "summary": f"Error: Invalid JSON input - {str(e)}",
            "model": "facebook/bart-large-cnn"
        }
        print(json.dumps(error_response))
        sys.exit(1)
    except Exception as e:
        error_response = {
            "summary": f"Error during processing: {str(e)}",
            "model": "facebook/bart-large-cnn"
        }
        print(json.dumps(error_response))
        sys.exit(1)

if __name__ == "__main__":
    main()