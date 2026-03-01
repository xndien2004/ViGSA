import regex as re

class VietnameseToneNormalizer:
    VOWELS_TABLE = [
        ['a', 'à', 'á', 'ả', 'ã', 'ạ', 'a'],
        ['ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ', 'aw'],
        ['â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ', 'aa'],
        ['e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ', 'e' ],
        ['ê', 'ề', 'ế', 'ể', 'ễ', 'ệ', 'ee'],
        ['i', 'ì', 'í', 'ỉ', 'ĩ', 'ị', 'i' ],
        ['o', 'ò', 'ó', 'ỏ', 'õ', 'ọ', 'o' ],
        ['ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ', 'oo'],
        ['ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ', 'ow'],
        ['u', 'ù', 'ú', 'ủ', 'ũ', 'ụ', 'u' ],
        ['ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự', 'uw'],
        ['y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ', 'y']
    ]
    
    # VOWELS_TO_IDS = {}
    # for i, row in enumerate(VOWELS_TABLE):
    #     for j, char in enumerate(row[:-1]):
    #         VOWELS_TO_IDS[char] = (i, j)
    VOWELS_TO_IDS = {
        'a': (0, 0), 'à': (0, 1), 'á': (0, 2), 'ả': (0, 3), 'ã': (0, 4), 'ạ': (0, 5), 
        'ă': (1, 0), 'ằ': (1, 1), 'ắ': (1, 2), 'ẳ': (1, 3), 'ẵ': (1, 4), 'ặ': (1, 5), 
        'â': (2, 0), 'ầ': (2, 1), 'ấ': (2, 2), 'ẩ': (2, 3), 'ẫ': (2, 4), 'ậ': (2, 5), 
        'e': (3, 0), 'è': (3, 1), 'é': (3, 2), 'ẻ': (3, 3), 'ẽ': (3, 4), 'ẹ': (3, 5), 
        'ê': (4, 0), 'ề': (4, 1), 'ế': (4, 2), 'ể': (4, 3), 'ễ': (4, 4), 'ệ': (4, 5), 
        'i': (5, 0), 'ì': (5, 1), 'í': (5, 2), 'ỉ': (5, 3), 'ĩ': (5, 4), 'ị': (5, 5), 
        'o': (6, 0), 'ò': (6, 1), 'ó': (6, 2), 'ỏ': (6, 3), 'õ': (6, 4), 'ọ': (6, 5), 
        'ô': (7, 0), 'ồ': (7, 1), 'ố': (7, 2), 'ổ': (7, 3), 'ỗ': (7, 4), 'ộ': (7, 5), 
        'ơ': (8, 0), 'ờ': (8, 1), 'ớ': (8, 2), 'ở': (8, 3), 'ỡ': (8, 4), 'ợ': (8, 5), 
        'u': (9, 0), 'ù': (9, 1), 'ú': (9, 2), 'ủ': (9, 3), 'ũ': (9, 4), 'ụ': (9, 5), 
        'ư': (10, 0), 'ừ': (10, 1), 'ứ': (10, 2), 'ử': (10, 3), 'ữ': (10, 4), 'ự': (10, 5), 
        'y': (11, 0), 'ỳ': (11, 1), 'ý': (11, 2), 'ỷ': (11, 3), 'ỹ': (11, 4), 'ỵ': (11, 5)
    }
    
    VINAI_NORMALIZED_TONE = {
        'òa': 'oà', 'Òa': 'Oà', 'ÒA': 'OÀ', 
        'óa': 'oá', 'Óa': 'Oá', 'ÓA': 'OÁ', 
        'ỏa': 'oả', 'Ỏa': 'Oả', 'ỎA': 'OẢ',
        'õa': 'oã', 'Õa': 'Oã', 'ÕA': 'OÃ',
        'ọa': 'oạ', 'Ọa': 'Oạ', 'ỌA': 'OẠ',
        'òe': 'oè', 'Òe': 'Oè', 'ÒE': 'OÈ',
        'óe': 'oé', 'Óe': 'Oé', 'ÓE': 'OÉ',
        'ỏe': 'oẻ', 'Ỏe': 'Oẻ', 'ỎE': 'OẺ',
        'õe': 'oẽ', 'Õe': 'Oẽ', 'ÕE': 'OẼ',
        'ọe': 'oẹ', 'Ọe': 'Oẹ', 'ỌE': 'OẸ',
        'ùy': 'uỳ', 'Ùy': 'Uỳ', 'ÙY': 'UỲ',
        'úy': 'uý', 'Úy': 'Uý', 'ÚY': 'UÝ',
        'ủy': 'uỷ', 'Ủy': 'Uỷ', 'ỦY': 'UỶ',
        'ũy': 'uỹ', 'Ũy': 'Uỹ', 'ŨY': 'UỸ',
        'ụy': 'uỵ', 'Ụy': 'Uỵ', 'ỤY': 'UỴ',
    }


    @staticmethod
    def normalize_unicode(text):
        char1252 = r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'
        charutf8 = r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'
        char_map = dict(zip(char1252.split('|'), charutf8.split('|')))
        return re.sub(char1252, lambda x: char_map[x.group()], text.strip())
    
    
    @staticmethod
    def normalize_sentence_typing(text, vinai_normalization=False):
        # https://github.com/VinAIResearch/BARTpho/blob/main/VietnameseToneNormalization.md
        if vinai_normalization: # Just simply replace the wrong tone with the correct one defined by VinAI
            for wrong, correct in VietnameseToneNormalizer.VINAI_NORMALIZED_TONE.items():
                text = text.replace(wrong, correct)
            return text.strip()
        
        # Or you can use this algorithm developed by Behitek to normalize Vietnamese typing in a sentence 
        words = text.strip().split()
        for index, word in enumerate(words):
            cw = re.sub(r'(^\p{P}*)([p{L}.]*\p{L}+)(\p{P}*$)', r'\1/\2/\3', word).split('/')
            if len(cw) == 3: cw[1] = VietnameseToneNormalizer.normalize_word_typing(cw[1])
            words[index] = ''.join(cw)
        return ' '.join(words)
    
     
    @staticmethod
    def normalize_word_typing(word):
        if not VietnameseToneNormalizer.is_valid_vietnamese_word(word): return word
        chars, vowel_indexes = list(word), []
        qu_or_gi, tonal_mark = False, 0
        
        for index, char in enumerate(chars):
            if char not in VietnameseToneNormalizer.VOWELS_TO_IDS: continue
            row, col = VietnameseToneNormalizer.VOWELS_TO_IDS[char]
            if index > 0 and (row, chars[index - 1]) in [(9, 'q'), (5, 'g')]:
                chars[index] = VietnameseToneNormalizer.VOWELS_TABLE[row][0]
                qu_or_gi = True
                
            if not qu_or_gi or index != 1: vowel_indexes.append(index)
            if col != 0:
                tonal_mark = col
                chars[index] = VietnameseToneNormalizer.VOWELS_TABLE[row][0]
                
        if len(vowel_indexes) < 2:
            if qu_or_gi:
                index = 1 if len(chars) == 2 else 2
                if chars[index] in VietnameseToneNormalizer.VOWELS_TO_IDS:
                    row, _ = VietnameseToneNormalizer.VOWELS_TO_IDS[chars[index]]
                    chars[index] = VietnameseToneNormalizer.VOWELS_TABLE[row][tonal_mark]
                else: chars[1] = VietnameseToneNormalizer.VOWELS_TABLE[5 if chars[1] == 'i' else 9][tonal_mark]
                return ''.join(chars)
            return word
        
        for index in vowel_indexes:
            row, _ = VietnameseToneNormalizer.VOWELS_TO_IDS[chars[index]]
            if row in [4, 8]: # ê, ơ
                chars[index] = VietnameseToneNormalizer.VOWELS_TABLE[row][tonal_mark]
                return ''.join(chars)
            
        index = vowel_indexes[0 if len(vowel_indexes) == 2 and vowel_indexes[-1] == len(chars) - 1 else 1] 
        row, _ = VietnameseToneNormalizer.VOWELS_TO_IDS[chars[index]]
        chars[index] = VietnameseToneNormalizer.VOWELS_TABLE[row][tonal_mark]
        return ''.join(chars)
    
    
    @staticmethod
    def is_valid_vietnamese_word(word):
        vowel_indexes = -1 
        for index, char in enumerate(word):
            if char not in VietnameseToneNormalizer.VOWELS_TO_IDS: continue
            if vowel_indexes in [-1, index - 1]: vowel_indexes = index
            else: return False
        return True