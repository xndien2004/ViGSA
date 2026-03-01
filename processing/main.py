from .data_parser import ParserData
from .preprocessor import VietnameseTextPreprocessor


class DataProcessor:
    def __init__(self, train_txt_path, val_txt_path=None, test_txt_path=None, max_length=512, language='vi'):
        self.train_txt_path = train_txt_path
        self.val_txt_path = val_txt_path
        self.test_txt_path = test_txt_path
        self.max_length = max_length

        self.vn_preprocessor = VietnameseTextPreprocessor(extra_teencodes={
            'khách sạn': ['ks'], 'nhà hàng': ['nhahang'], 'nhân viên': ['nv'],
            'cửa hàng': ['store', 'sop', 'shopE', 'shop'],
            'sản phẩm': ['sp', 'product'], 'hàng': ['hàg'],
            'giao hàng': ['ship', 'delivery', 'síp'], 'đặt hàng': ['order'],
            'chuẩn chính hãng': ['authentic', 'aut', 'auth'], 'hạn sử dụng': ['date', 'hsd'],
            'điện thoại': ['dt'],  'facebook': ['fb', 'face'],
            'nhắn tin': ['nt', 'ib'], 'trả lời': ['tl', 'trl', 'rep'],
            'feedback': ['fback', 'fedback'], 'sử dụng': ['sd'], 'xài': ['sài'],
        }, max_correction_length=self.max_length, path=train_txt_path)

    def process(self, text_column):
        self.parser = ParserData(
            train_txt_path=self.train_txt_path,
            val_txt_path=self.val_txt_path,
            test_txt_path=self.test_txt_path
        )
        dataframe =  self.parser.to_dataframes() 
        for key, df in dataframe.items():
            df[text_column] = df[text_column].apply(self.vn_preprocessor.process_text)
            dataframe[key] = df

        return dataframe
