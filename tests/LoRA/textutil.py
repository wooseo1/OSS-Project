"""텍스트 정규화, CER/WER 계산, 자모(음소) 단위 오류 분석.

외부 라이브러리 없이 동작함 (학습/평가 스크립트에서 공통 사용).
"""
import re
import unicodedata
from collections import Counter, defaultdict

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize(text):
    """평가용 정규화: NFC, 소문자, 구두점 제거, 공백 정리"""
    text = unicodedata.normalize("NFC", text or "")
    text = _PUNCT.sub(" ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(a, b):
    """두 시퀀스(문자열/리스트)의 레벤슈타인 거리"""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def cer_counts(ref, hyp):
    """(오류 수, 정답 글자 수). 공백 제외"""
    r = normalize(ref).replace(" ", "")
    h = normalize(hyp).replace(" ", "")
    return edit_distance(r, h), len(r)


def wer_counts(ref, hyp):
    """(오류 수, 정답 단어 수). 공백 기준 토큰"""
    r = normalize(ref).split()
    h = normalize(hyp).split()
    return edit_distance(r, h), len(r)


def corpus_rates(pairs):
    """(ref, hyp) 쌍 목록 -> (CER, WER). 전체 합산 방식"""
    ce = cn = we = wn = 0
    for ref, hyp in pairs:
        e, n = cer_counts(ref, hyp)
        ce += e
        cn += n
        e, n = wer_counts(ref, hyp)
        we += e
        wn += n
    return (ce / cn if cn else 0.0, we / wn if wn else 0.0)


# ---------------------------------------------------------------- 자모 분해
_CHO = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
_JUNG = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
_JONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ",
         "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ",
         "ㅍ", "ㅎ"]


def to_jamo_tokens(text):
    """한글 음절을 위치 표시가 붙은 자모 토큰으로 분해.

    예) '강' -> ['초ㄱ', '중ㅏ', '종ㅇ']
    초성 ㅇ(소리 없음)과 종성 ㅇ(받침 이응)을 구분하기 위해 위치(초/중/종)를 붙임.
    한글이 아닌 글자는 그대로 둠.
    """
    out = []
    for ch in normalize(text).replace(" ", ""):
        code = ord(ch) - 0xAC00
        if 0 <= code < 11172:
            out.append("초" + _CHO[code // 588])
            out.append("중" + _JUNG[(code % 588) // 28])
            t = _JONG[code % 28]
            if t:
                out.append("종" + t)
        else:
            out.append(ch)
    return out


def align(ref, hyp):
    """토큰 시퀀스 정렬. 반환: [(op, ref_tok, hyp_tok)], op: '=', 'S', 'D', 'I'"""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            ops.append(("=" if ref[i - 1] == hyp[j - 1] else "S", ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("D", ref[i - 1], ""))
            i -= 1
        else:
            ops.append(("I", "", hyp[j - 1]))
            j -= 1
    ops.reverse()
    return ops


class TokenErrorStats:
    """자모 토큰별 오류 통계 누적 (어떤 음소가 얼마나, 무엇으로 틀리는지)"""

    def __init__(self):
        self.total = Counter()
        self.sub = Counter()
        self.dele = Counter()
        self.ins = Counter()
        self.pairs = defaultdict(Counter)  # ref_tok -> Counter(hyp_tok)

    def add(self, ref_text, hyp_text):
        for op, r, h in align(to_jamo_tokens(ref_text), to_jamo_tokens(hyp_text)):
            if op == "=":
                self.total[r] += 1
            elif op == "S":
                self.total[r] += 1
                self.sub[r] += 1
                self.pairs[r][h] += 1
            elif op == "D":
                self.total[r] += 1
                self.dele[r] += 1
            else:
                self.ins[h] += 1

    def rows(self):
        """토큰별 요약 행 목록 (오류율 높은 순)"""
        rows = []
        for tok, n in self.total.items():
            errs = self.sub[tok] + self.dele[tok]
            top = ", ".join(f"{h}({c})" for h, c in self.pairs[tok].most_common(3))
            rows.append({
                "token": tok,
                "count": n,
                "errors": errs,
                "error_rate": round(errs / n, 4) if n else 0.0,
                "substitutions": self.sub[tok],
                "deletions": self.dele[tok],
                "top_confusions": top,
            })
        rows.sort(key=lambda r: (-r["error_rate"], -r["count"]))
        return rows

    def pair_rows(self):
        """ref -> hyp 대체 오류 쌍 (많은 순)"""
        rows = []
        for r, ctr in self.pairs.items():
            for h, c in ctr.items():
                rows.append({"ref": r, "hyp": h, "count": c})
        rows.sort(key=lambda x: -x["count"])
        return rows
