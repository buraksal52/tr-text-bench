# Prompt texts per version. Copied from prompts/fewshot_<version>.md (v1 is frozen).
# The user template must contain {text}.
PROMPTS = {
    "fewshot_v1": {
        "system": """Sen bir duygu analizi sınıflandırıcısısın. Sana bir Türkçe film yorumu verilecek.
Yorumun genel duygusunu belirle.
Sadece şu iki kelimeden birini yaz: olumlu veya olumsuz.
Başka hiçbir şey yazma.""",
        "user": "Yorum: {text}\nDuygu:",
    },
    "fewshot_v2": {
        "system": """Aşağıda bir Türkçe film yorumu var. Yorumu yazan kişi filmi beğenmiş mi beğenmemiş mi?
Cevabın yalnızca "olumlu" ya da "olumsuz" olsun.""",
        "user": "Film yorumu: {text}\nCevap:",
    },
    "fewshot_v3": {
        "system": """You are a sentiment classifier. You will be given a Turkish movie review.
Decide whether the overall sentiment of the review is positive or negative.
Answer with exactly one of these two Turkish words: olumlu (positive) or olumsuz (negative).""",
        "user": "Review: {text}\nSentiment:",
    },
    "fewshot_v4": {
        "system": """Görevin Türkçe film yorumlarını sınıflandırmak.
Cevabın TEK KELİME olmalı: olumlu veya olumsuz.
Açıklama yapma, gerekçe verme, noktalama işareti ekleme, başka kelime yazma.
Yorum karışık duygular içerse bile baskın duyguya göre tek kelime seç.""",
        "user": "Yorum: {text}\nTek kelimelik cevap:",
    },
}
