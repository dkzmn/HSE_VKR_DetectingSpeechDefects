import sqlite3

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup

BTN_SET_LETTERS = "Выбрать контрольные буквы"
BTN_GET_TWISTER = "Получить скороговорку"
BTN_MY_RESULTS = "Мои результаты"
BTN_GRANT_ACCESS = "Выдать доступ логопеду"
BTN_REVOKE_ACCESS = "Отозвать доступ у логопеда"
BTN_PATIENTS = "Мои пациенты"
BTN_PATIENT_RESULTS = "Результаты пациента"
BTN_SET_ROLE = "Назначить роль"
BTN_REVOKE_ROLE = "Отозвать роль логопеда"
BTN_HELP = "Помощь"
BTN_LETTERS_DONE = "Готово"

ALL_LETTERS = ["л", "р", "с", "т", "ц", "ч", "ш", "щ"]


def build_menu(role: str) -> ReplyKeyboardMarkup:
    # кнопки меню зависят от роли пользователя
    base_rows = [
        [BTN_SET_LETTERS, BTN_GET_TWISTER],
        [BTN_MY_RESULTS, BTN_HELP],
    ]
    if role == "patient":
        rows = [*base_rows, [BTN_GRANT_ACCESS, BTN_REVOKE_ACCESS]]
    elif role == "therapist":
        rows = [*base_rows, [BTN_PATIENTS, BTN_PATIENT_RESULTS]]
    else:
        rows = [*base_rows, [BTN_SET_ROLE, BTN_REVOKE_ROLE], [BTN_PATIENTS, BTN_PATIENT_RESULTS]]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)


def build_letters_keyboard(selected_letters: list[str]) -> InlineKeyboardMarkup:
    # inline-клавиатура с чекбоксами для каждой буквы и кнопкой «Готово»
    buttons = []
    selected = set(selected_letters)
    for letter in ALL_LETTERS:
        mark = "✅" if letter in selected else "⬜"
        buttons.append(
            InlineKeyboardButton(f"{mark} {letter.upper()}", callback_data=f"letters:toggle:{letter}")
        )
    rows = [[buttons[i], buttons[i + 1]] for i in range(0, len(buttons), 2)]
    rows.append([InlineKeyboardButton(BTN_LETTERS_DONE, callback_data="letters:done")])
    return InlineKeyboardMarkup(rows)


def format_letters_text(selected_letters: list[str]) -> str:
    # формирует строку с перечнем выбранных букв для отображения
    if not selected_letters:
        return "Выберите контрольные буквы:\nПока ничего не выбрано."
    return "Выберите контрольные буквы:\n " + ", ".join([x.upper() for x in selected_letters])


def build_therapists_keyboard(therapists: list[sqlite3.Row], action: str) -> InlineKeyboardMarkup:
    # inline-список логопедов с callback вида access:<action>:<id>
    rows = []
    for t in therapists:
        title = t["name"] or f"id:{t['telegram_id']}"
        rows.append([InlineKeyboardButton(title, callback_data=f"access:{action}:{t['id']}")])
    return InlineKeyboardMarkup(rows)


def build_users_keyboard(users: list[sqlite3.Row], action: str) -> InlineKeyboardMarkup:
    # inline-список пользователей с callback вида role:<action>:<id>
    rows = []
    for u in users:
        title = f"{u['name'] or 'Без ФИО'} ({u['role']})"
        rows.append([InlineKeyboardButton(title, callback_data=f"role:{action}:{u['id']}")])
    return InlineKeyboardMarkup(rows)
