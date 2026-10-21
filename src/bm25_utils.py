import re
from bm25s.tokenization import Tokenizer


def korean_splitter(text: str) -> list[str]:
    """
    영문/숫자 토큰과 한글 토큰을 분리한다.

    예)
    'HBM이란 무엇인가?'
    -> ['hbm', '이란', '무엇인가']

    'HBM은 고대역폭 메모리'
    -> ['hbm', '은', '고대역폭', '메모리']
    """
    return re.findall(r"[A-Za-z0-9]+|[가-힣]+", text.lower())


def create_bm25_tokenizer():
    return Tokenizer(
        splitter=korean_splitter,
        stopwords=[],
        stemmer=None,
    )
