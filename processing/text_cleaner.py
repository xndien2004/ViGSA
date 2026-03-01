import re
import emoji

class VietnameseTextCleaner: # https://ihateregex.io
    VN_CHARS = 'áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÍÌỈĨỊÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ'
    
    @staticmethod
    def remove_html(text):
        return re.sub(r'<[^>]*>', '', text)
    
    @staticmethod
    def remove_emoji(text):
        return emoji.replace_emoji(text, '')
    
    @staticmethod
    def remove_url(text):
        return re.sub(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()!@:%_\+.~#?&\/\/=]*)', '', text)
    
    @staticmethod
    def remove_email(text):
        return re.sub(r'[^@ \t\r\n]+@[^@ \t\r\n]+\.[^@ \t\r\n]+', '', text)
    
    @staticmethod
    def remove_phone_number(text):
        return re.sub(r'^[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}$', '', text)
    
    @staticmethod
    def remove_hashtags(text):
        return re.sub(r'#\w+', '', text)
    
    @staticmethod
    def remove_unnecessary_characters(text):
        text = re.sub(fr"[^\sa-zA-Z0-9{VietnameseTextCleaner.VN_CHARS}]", ' ', text)
        return re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
    
    @staticmethod
    def process_text(text):
        text = VietnameseTextCleaner.remove_html(text)
        # text = VietnameseTextCleaner.remove_emoji(text)
        text = VietnameseTextCleaner.remove_url(text)
        text = VietnameseTextCleaner.remove_email(text)
        text = VietnameseTextCleaner.remove_phone_number(text)
        text = VietnameseTextCleaner.remove_hashtags(text)
        # text = VietnameseTextCleaner.remove_unnecessary_characters(text)
        return text
