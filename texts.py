"""Parallel test corpus for Lab 01.

The same three items in English, Russian and Kazakh. Parallel meaning is the
point: any difference in token count is a property of the tokenizer, not of
what is being said.

Instructors: the Kazakh and Russian wordings are a starting point. Substitute
your own if you prefer -- but keep the three versions semantically parallel,
otherwise the comparison measures translation length instead of tokenization.
"""

from __future__ import annotations

from typing import Dict

LANGUAGES = ("en", "ru", "kk")

#: One sentence. Short enough to inspect token by token.
SENTENCE: Dict[str, str] = {
    "en": "The bank raised interest rates by two percentage points last quarter.",
    "ru": "Банк повысил процентные ставки на два процентных пункта в прошлом квартале.",
    "kk": "Банк өткен тоқсанда пайыздық мөлшерлемені екі пайыздық тармаққа көтерді.",
}

#: A realistic support request -- the kind of text a production system pays for
#: thousands of times a day.
COMPLAINT: Dict[str, str] = {
    "en": (
        "Good afternoon. I opened a deposit at your branch in March and was told "
        "the rate was fixed for twelve months. In August the rate on my account "
        "dropped without any notice. I have attached the contract and the "
        "statement. Please explain on what basis the rate was changed and "
        "restore the original terms."
    ),
    "ru": (
        "Добрый день. Я открыл депозит в вашем отделении в марте, и мне сказали, "
        "что ставка зафиксирована на двенадцать месяцев. В августе ставка по "
        "моему счёту снизилась без какого-либо уведомления. Прилагаю договор и "
        "выписку. Прошу объяснить, на каком основании была изменена ставка, и "
        "восстановить первоначальные условия."
    ),
    "kk": (
        "Қайырлы күн. Мен наурыз айында сіздің бөлімшеңізде депозит аштым, маған "
        "мөлшерлеме он екі айға бекітілген деп айтылды. Тамыз айында менің "
        "шотымдағы мөлшерлеме ешқандай хабарламасыз төмендеді. Шартты және "
        "үзінді көшірмені қоса тіркеп отырмын. Мөлшерлеме қандай негізде "
        "өзгертілгенін түсіндіріп, бастапқы шарттарды қалпына келтіруіңізді "
        "сұраймын."
    ),
}

#: A system prompt -- the part you resend on every single request.
SYSTEM_PROMPT: Dict[str, str] = {
    "en": (
        "You are a support assistant for a retail bank. Answer only from the "
        "documents provided. If the answer is not in them, say so. Never invent "
        "an account number, a rate or a date."
    ),
    "ru": (
        "Вы — ассистент поддержки розничного банка. Отвечайте только по "
        "предоставленным документам. Если ответа в них нет, так и скажите. "
        "Никогда не выдумывайте номер счёта, ставку или дату."
    ),
    "kk": (
        "Сіз — бөлшек банктің қолдау көрсету ассистентісіз. Тек берілген "
        "құжаттар бойынша жауап беріңіз. Егер жауап оларда болмаса, солай деп "
        "айтыңыз. Шот нөмірін, мөлшерлемені немесе күнді ешқашан ойдан "
        "шығармаңыз."
    ),
}

#: Everything the lab measures, keyed by a short id.
CORPUS: Dict[str, Dict[str, str]] = {
    "sentence": SENTENCE,
    "complaint": COMPLAINT,
    "system_prompt": SYSTEM_PROMPT,
}
