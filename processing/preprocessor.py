import os
import requests
from io import StringIO
from underthesea import word_tokenize
from transformers import pipeline
from .text_cleaner import VietnameseTextCleaner
from .tone_normalizer import VietnameseToneNormalizer

class VietnameseTextPreprocessor:
    def __init__(self, extra_teencodes=None, max_correction_length=512, path="uit"):
        self.extra_teencodes = extra_teencodes
        self._build_teencodes()
        
        self.max_correction_length = max_correction_length
        self.path = path
        self.corrector = pipeline(
            'text-generation', model='bmd1905/vietnamese-correction-v2', 
            torch_dtype='bfloat16', device_map='auto', num_workers=os.cpu_count()
        )
        print('bmd1905/vietnamese-correction-v2 is loaded successfully.')
    
    def _build_teencodes(self):
        self.teencodes = {
            'ok': ['okie', 'okey', 'ôkê', 'oki', 'oke', 'okay', 'okê'], 
            'không': ['kg', 'not', 'k', 'kh', 'kô', 'hok', 'ko', 'khong'], 
            'không phải': ['kp'], 
            'cảm ơn': ['tks', 'thks', 'thanks', 'ths', 'thank'], 
            'hồi đó': ['hùi đó'], 
            'muốn': ['mún'],
            'thôi': ['thui'],
            'giả mạo': ['fake'], 
            'thích': ['thik'], 
            'yêu': ['iu'], 
            'với': ['vs'], 
            'gì': ['j'], 
            'rồi': ['r'], 
            'mình': ['m', 'mik'], 
            'giờ': ['h'], 
            'thời gian': ['time'],
            'được': ['đx', 'dk', 'dc', 'đk', 'đc'], 
            'bình thường': ['bt', 'bthg'], 
            'hàng': ['hàg'], 

            'rất tốt': ['perfect', '❤️', '😍', '💘', '💓', '💖', '💝', '💕', '💞', '😘'], 
            'tốt': [
                'gud', 'good', 'gút', 'tot', 'nice', 'thick', '👍', '🎉', '😀', '😂', '🤗',
                '😙', '🙂', '^_^', ':)', '=)', 'hehe', 'hihi', 'haha', 'hjhj', '😋', '😄', '😆', '😎'
            ],
            'dễ thương': ['cute'],
            'quá': ['wa', 'wá', 'qá'],

            'không tốt': ['lol', 'cc', 'huhu', ':(', '😔', '😓', '😤', '😭', '😑', '😳', '😥', '🤔'],
            'tệ': ['sad', 'por', 'poor', 'bad'], 
        }

        if self.extra_teencodes: 
            for key, values in self.extra_teencodes.items():
                if any(len(value.split()) > 1 for value in values):
                    raise ValueError('The values for each key in extra_teencodes must be single words.')
                self.teencodes.setdefault(key, []).extend(values)
                
        self.teencodes = {word: key for key, values in self.teencodes.items() for word in values}
        teencode_url = 'https://raw.githubusercontent.com/htuann2712/ABSA-VLSP2018/refs/heads/main/teencode.txt'
        response = requests.get(teencode_url)
        
        if response.status_code == 200:
            text_data = StringIO(response.text)
            for pair in text_data:
                teencode, true_text = pair.split('\t')
                self.teencodes[teencode.strip()] = true_text.strip()
            self.teencodes = {k: self.teencodes[k] for k in sorted(self.teencodes)}
        else: print('Failed to fetch teencode.txt from', teencode_url)
    
    def normalize_teencodes(self, text):
        words = []
        for word in text.split():
            words.append(self.teencodes.get(word, word))
        return ' '.join(words)
    
    def correct_vietnamese_errors(self, texts):
        # https://huggingface.co/bmd1905/vietnamese-correction-v2
        predictions = self.corrector(texts, max_length=self.max_correction_length, truncation=True)
        return [prediction['generated_text'] for prediction in predictions]
    
    def word_segment(self, text):
        # Use underthesea for word segmentation
        return word_tokenize(text, format='text')
    
    def process_text(self, text, normalize_tone=True, segment=True):
        text = text.lower()
        if "vlsp" in self.path.lower():
            text = VietnameseTextCleaner.process_text(text)
        # text = VietnameseTextCleaner.process_text(text)
        text = self.normalize_teencodes(text)
        if normalize_tone and "vlsp" in self.path.lower():
            text = VietnameseToneNormalizer.normalize_unicode(text)
            text = VietnameseToneNormalizer.normalize_sentence_typing(text)
        return self.word_segment(text) if segment else text
    
    def process_batch(self, texts, correct_errors=True):
        if correct_errors:
            texts = [self.process_text(text, normalize_tone=True, segment=False) for text in texts]
            texts = self.correct_vietnamese_errors(texts)
            return [self.process_text(text, normalize_tone=False, segment=True) for text in texts]
        return [self.process_text(text, normalize_tone=True, segment=True) for text in texts]
    