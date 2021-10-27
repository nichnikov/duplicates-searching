import re
from pymystem3 import Mystem


class TextsLematizer:
    def __init__(self):
        self.m = Mystem()

    def text_lemmatize(self, text: str):
        """метод лемматизации одного текста"""
        try:
            lemm_txt = self.m.lemmatize(text)
            lemm_txt = [w for w in lemm_txt if w not in [' ', '\n', ' \n']]
            return lemm_txt
        except:
            return ['']

    def texts_lemmatize(self, texts_list):
        """функция лемматизации списка текстов текста"""
        return [self.text_lemmatize(text_hangling(tx)) for tx in texts_list]

    def __call__(self, texts_list):
        return self.texts_lemmatize(texts_list)


texts_lemmatize = TextsLematizer()


def codes_parsing(text: str) -> str:
    """Создание и имплементация в проект "Быстрых ответов" правил, обрабатывающих числовые последовательности типа:
        1111.11.11 и 1111.11.11-11 для одного текста"""
    pttrn = r"\d+\.\d+\.\d+\.\d+\-\d+|\d+\.\d+\.\d+\-\d+|\d+\.\d+\-\d+|\d+\.\d+\.\d+\.\d+|\d+\.\d+\.\d+|\d+\.\d+"
    finded = re.findall(pttrn, text)
    try:
        if finded:
            for dgt in finded:
                dgt_ = re.sub(r"[^\w]", "", str("".join(dgt)))
                text = re.sub(str(dgt), dgt_, text)
            return text
        else:
            return text
    except:
        return text


def text_hangling(text: str) -> str:
    """функция, проводящая предобработку текста"""
    try:
        txt = codes_parsing(text)
        txt = re.sub(r'[^a-zа-я\d]', ' ', txt.lower())
        txt = re.sub(r'\s+', ' ', txt)
        return txt
    except:
        return ""


def asc_dsc_apply(asc_dsc_list: [], texts_list: []) -> []:
    """применение аскрипторов и дескрипторов (из списка аскрипторов-дескрипторов) к списку текстов
    возвращает список текстов, с заменой синонимов и биграмм"""
    texts = '\n'.join(texts_list)
    asc_dsc_list_ = [(" " + x + " ", " " + y + " ") for x, y in asc_dsc_list]
    asc_dsc_list_ = [(re.sub(r"\s+", " ", x), re.sub(r"\s+", " ", y)) for x, y in asc_dsc_list_]
    for asc, dsc in asc_dsc_list_:
        # texts = re.sub(asc, dsc, texts)
        texts = texts.replace(asc, dsc)
    return texts.split('\n')


class SimpleTokenizerFast:
    """класс аналогичный SimpleTokenizer, но токенизирует другим алгоритмом (более быстрым, разница особенно
    существенная на большой текстовой коллекции (на порядки)"""

    def __init__(self, lingv_parametrs: {}):
        if "asc_dsc_list" in lingv_parametrs:
            self.asc_dsc_list = lingv_parametrs["asc_dsc_list"]
        else:
            self.asc_dsc_list = []
        if "stopwords" in lingv_parametrs:
            self.stopwords = lingv_parametrs["stopwords"]
        else:
            self.stopwords = []
        if "keywords" in lingv_parametrs:
            self.keywords = lingv_parametrs["keywords"]
        else:
            self.keywords = []

    def texts_processing(self, texts_list: []) -> []:
        """функция, лемматизирующая входящий список текстов и применяющая к ним лингвистику:
        синонимы, биграммы, стоп-слова, возвращает список списков лемматизированных токенов"""
        tz_txs = asc_dsc_apply(self.asc_dsc_list, [" ".join(tx) for tx in texts_lemmatize(texts_list)])
        tz_txs_split = [tx.split() for tx in tz_txs]
        if self.keywords:
            return [[w for w in tx if w not in self.stopwords and w in self.keywords] for tx in tz_txs_split]
        else:
            return [[w for w in tx if w not in self.stopwords] for tx in tz_txs_split]

    def __call__(self, texts_list):
        return self.texts_processing(texts_list)
