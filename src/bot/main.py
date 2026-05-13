import json
import os
import subprocess
from datetime import datetime

import requests
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from db import (
    AUDIO_DIR,
    LETTER_TO_COL,
    db_conn,
    get_or_create_user,
    get_selected_letters,
    init_db,
    pick_twister,
)
from keyboards import (
    ALL_LETTERS,
    BTN_GET_TWISTER,
    BTN_GRANT_ACCESS,
    BTN_HELP,
    BTN_MY_RESULTS,
    BTN_PATIENT_RESULTS,
    BTN_PATIENTS,
    BTN_REVOKE_ACCESS,
    BTN_REVOKE_ROLE,
    BTN_SET_LETTERS,
    BTN_SET_ROLE,
    build_letters_keyboard,
    build_menu,
    build_therapists_keyboard,
    build_users_keyboard,
    format_letters_text,
)

ML_API_URL = os.getenv("ML_API_URL", "http://ml_api:8000/score")
BOT_TOKEN = os.getenv("BOT_TOKEN", "")


async def require_name(update: Update, user) -> bool:
    # блокирует действие и просит ввести ФИО, если оно не указано
    if user["name"] and user["name"].strip():
        return True
    await update.message.reply_text("Для начала работы введите ваше ФИО одним сообщением.")
    return False


async def show_letters_picker(target_message, user):
    # отправляет сообщение с inline-клавиатурой выбора букв
    letters = get_selected_letters(user)
    await target_message.reply_text(
        format_letters_text(letters),
        reply_markup=build_letters_keyboard(letters),
    )


async def show_grant_access_picker(target_message):
    # показывает список логопедов, которым пациент может выдать доступ
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, telegram_id, name FROM users WHERE role IN ('therapist', 'admin') ORDER BY name"
    )
    therapists = cur.fetchall()
    conn.close()
    if not therapists:
        await target_message.reply_text("Пока нет пользователей с ролью логопеда.")
        return
    await target_message.reply_text(
        "Выберите логопеда для выдачи доступа:",
        reply_markup=build_therapists_keyboard(therapists, "grant"),
    )


async def show_revoke_access_picker(target_message, patient_id: int):
    # показывает только тех логопедов, которым уже выдан доступ
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT u.id, u.telegram_id, u.name
        FROM patient_therapist_access pta
        JOIN users u ON u.id = pta.therapist_id
        WHERE pta.patient_id = ?
        ORDER BY u.name
        """,
        (patient_id,),
    )
    therapists = cur.fetchall()
    conn.close()
    if not therapists:
        await target_message.reply_text("У вас нет логопедов с выданным доступом.")
        return
    await target_message.reply_text(
        "Выберите логопеда для отзыва доступа:",
        reply_markup=build_therapists_keyboard(therapists, "revoke"),
    )


async def show_set_role_picker(target_message):
    # показывает пациентов, которых можно назначить логопедом
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, telegram_id, name, role
        FROM users
        WHERE role NOT IN ('therapist', 'admin')
        ORDER BY id DESC
        LIMIT 20
        """
    )
    users = cur.fetchall()
    conn.close()
    if not users:
        await target_message.reply_text("Нет пользователей для назначения роли логопеда.")
        return
    await target_message.reply_text(
        "Выберите пользователя, которого сделать логопедом:",
        reply_markup=build_users_keyboard(users, "set_therapist"),
    )


async def show_revoke_role_picker(target_message):
    # показывает текущих логопедов для отзыва роли
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, telegram_id, name, role FROM users WHERE role = 'therapist' ORDER BY name"
    )
    users = cur.fetchall()
    conn.close()
    if not users:
        await target_message.reply_text("Нет логопедов для отзыва роли.")
        return
    await target_message.reply_text(
        "Выберите логопеда для отзыва роли (вернётся в статус пациента):",
        reply_markup=build_users_keyboard(users, "revoke_therapist"),
    )


async def show_patient_results_picker(target_message, therapist_db_id: int):
    # показывает inline-список пациентов, к которым логопед имеет доступ
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT u.id, u.name
        FROM patient_therapist_access pta
        JOIN users u ON u.id = pta.patient_id
        WHERE pta.therapist_id = ?
        ORDER BY u.name
        """,
        (therapist_db_id,),
    )
    patients = cur.fetchall()
    conn.close()
    if not patients:
        await target_message.reply_text("Нет пациентов с доступом.")
        return
    rows = [
        [InlineKeyboardButton(p["name"] or f"id:{p['id']}", callback_data=f"patient:results:{p['id']}")]
        for p in patients
    ]
    await target_message.reply_text(
        "Выберите пациента:",
        reply_markup=InlineKeyboardMarkup(rows),
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # приветственное сообщение со списком команд и меню по роли
    user = get_or_create_user(update)
    if not await require_name(update, user):
        return
    await update.message.reply_text(
        "Привет! Используйте кнопки меню ниже.\n"
        "Текстовые команды:\n"
        "/set_name <ФИО>\n"
        "/set_letters л р ш\n"
        "/get_twister\n"
        "/grant_access <telegram_id_логопеда>\n"
        "/revoke_access <telegram_id_логопеда>\n"
        "/my_results\n"
        "/patients (для логопеда)\n"
        "/patient_results <telegram_id_пациента> (для логопеда)\n"
        "/set_role <telegram_id> <admin|therapist|patient> (для admin)\n"
        f"Ваша роль: {user['role']}",
        reply_markup=build_menu(user["role"]),
    )


async def set_name(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # сохраняет ФИО пользователя
    user = get_or_create_user(update)
    if not context.args:
        current_name = user["name"] or "не указано"
        await update.message.reply_text(
            f"Текущее ФИО: {current_name}\nИспользование: /set_name <ФИО полностью>"
        )
        return
    full_name = " ".join(context.args).strip()
    if len(full_name.split()) < 2:
        await update.message.reply_text("Пожалуйста, введите ФИО полностью (минимум имя и фамилия).")
        return
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("UPDATE users SET name = ? WHERE telegram_id = ?", (full_name, update.effective_user.id))
    conn.commit()
    conn.close()
    await update.message.reply_text(f"ФИО обновлено: {full_name}", reply_markup=build_menu(user["role"]))


async def handle_menu_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # маршрутизирует нажатия кнопок ReplyKeyboard к соответствующим обработчикам
    text = (update.message.text or "").strip()
    user = get_or_create_user(update)
    if not (user["name"] and user["name"].strip()):
        if len(text.split()) < 2:
            await update.message.reply_text("Пожалуйста, введите ФИО полностью (минимум имя и фамилия).")
            return
        conn = db_conn()
        cur = conn.cursor()
        cur.execute("UPDATE users SET name = ? WHERE telegram_id = ?", (text, update.effective_user.id))
        conn.commit()
        conn.close()
        await update.message.reply_text(f"Спасибо! ФИО сохранено: {text}", reply_markup=build_menu(user["role"]))
        return

    if text == BTN_HELP:
        await start(update, context)
        return
    if text == BTN_GET_TWISTER:
        await get_twister(update, context)
        return
    if text == BTN_MY_RESULTS:
        await my_results(update, context)
        return
    if text == BTN_PATIENTS:
        await patients(update, context)
        return
    if text == BTN_SET_LETTERS:
        await show_letters_picker(update.message, user)
        return
    if text == BTN_GRANT_ACCESS:
        await show_grant_access_picker(update.message)
        return
    if text == BTN_REVOKE_ACCESS:
        await show_revoke_access_picker(update.message, user["id"])
        return
    if text == BTN_PATIENT_RESULTS:
        if user["role"] not in {"therapist", "admin"}:
            await update.message.reply_text("Команда только для логопеда или админа")
            return
        await show_patient_results_picker(update.message, user["id"])
        return
    if text == BTN_SET_ROLE:
        if user["role"] != "admin":
            await update.message.reply_text("Только admin")
            return
        await show_set_role_picker(update.message)
        return
    if text == BTN_REVOKE_ROLE:
        if user["role"] != "admin":
            await update.message.reply_text("Только admin")
            return
        await show_revoke_role_picker(update.message)
        return


async def set_role(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # команда admin: изменить роль пользователя по telegram_id
    actor = get_or_create_user(update)
    if not await require_name(update, actor):
        return
    if actor["role"] != "admin":
        await update.message.reply_text("Только admin")
        return
    if len(context.args) == 0:
        await show_set_role_picker(update.message)
        return
    if len(context.args) != 2:
        await update.message.reply_text("Использование: /set_role <telegram_id> <role>")
        return
    tg_id, role = context.args[0], context.args[1]
    if not tg_id.isdigit() or role not in {"admin", "therapist", "patient"}:
        await update.message.reply_text("Некорректные данные")
        return
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT role FROM users WHERE telegram_id = ?", (int(tg_id),))
    target = cur.fetchone()
    if not target:
        conn.close()
        await update.message.reply_text("Пользователь не найден. Пусть сначала напишет /start")
        return
    if target["role"] == "admin" and role == "therapist":
        conn.close()
        await update.message.reply_text("Админа нельзя сделать логопедом")
        return
    cur.execute("UPDATE users SET role = ? WHERE telegram_id = ?", (role, int(tg_id)))
    conn.commit()
    changed = cur.rowcount
    conn.close()
    if changed:
        await update.message.reply_text("Роль обновлена")
    else:
        await update.message.reply_text("Пользователь не найден. Пусть сначала напишет /start")


async def set_letters(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # сохраняет выбранные контрольные буквы для фильтрации скороговорок
    user = get_or_create_user(update)
    if not await require_name(update, user):
        return
    if not context.args:
        await show_letters_picker(update.message, user)
        return
    letters = [x.lower() for x in context.args]
    bad = [x for x in letters if x not in LETTER_TO_COL]
    if bad:
        await update.message.reply_text("Допустимые буквы: л р с т ц ч ш щ")
        return
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE users SET selected_letters = ? WHERE telegram_id = ?",
        (json.dumps(letters, ensure_ascii=False), update.effective_user.id),
    )
    conn.commit()
    conn.close()
    await update.message.reply_text(f"Выбраны буквы: {', '.join(letters) if letters else 'пусто'}")


async def handle_letters_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # переключает выбор буквы или сохраняет результат по нажатию «Готово»
    query = update.callback_query
    await query.answer()
    user = get_or_create_user(update)
    if not await require_name(update, user):
        return
    data = query.data or ""
    if data == "letters:done":
        letters = get_selected_letters(user)
        text = "Выбор сохранен. Текущие буквы: " + (", ".join(letters) if letters else "пусто")
        await query.edit_message_text(text=text)
        return
    if not data.startswith("letters:toggle:"):
        return
    letter = data.split(":")[-1].lower()
    if letter not in LETTER_TO_COL:
        return
    letters = set(get_selected_letters(user))
    if letter in letters:
        letters.remove(letter)
    else:
        letters.add(letter)
    updated = sorted(list(letters), key=ALL_LETTERS.index)
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE users SET selected_letters = ? WHERE telegram_id = ?",
        (json.dumps(updated, ensure_ascii=False), update.effective_user.id),
    )
    conn.commit()
    conn.close()
    await query.edit_message_text(
        text=format_letters_text(updated),
        reply_markup=build_letters_keyboard(updated),
    )


async def handle_access_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # выдаёт или отзывает доступ логопеда к данным пациента
    query = update.callback_query
    await query.answer()
    user = get_or_create_user(update)
    if not await require_name(update, user):
        return
    _, action, therapist_id_s = (query.data or "").split(":")
    therapist_id = int(therapist_id_s)
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, name FROM users WHERE id = ? AND role IN ('therapist', 'admin')",
        (therapist_id,),
    )
    therapist = cur.fetchone()
    if not therapist:
        conn.close()
        await query.edit_message_text("Логопед не найден.")
        return
    if action == "grant":
        cur.execute(
            "INSERT OR IGNORE INTO patient_therapist_access (patient_id, therapist_id) VALUES (?, ?)",
            (user["id"], therapist_id),
        )
        conn.commit()
        conn.close()
        await query.edit_message_text(f"Доступ выдан логопеду: {therapist['name'] or therapist_id}")
        return
    if action == "revoke":
        cur.execute(
            "DELETE FROM patient_therapist_access WHERE patient_id = ? AND therapist_id = ?",
            (user["id"], therapist_id),
        )
        conn.commit()
        conn.close()
        await query.edit_message_text(f"Доступ отозван у логопеда: {therapist['name'] or therapist_id}")
        return
    conn.close()


async def handle_role_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # назначает или отзывает роль логопеда (только admin)
    query = update.callback_query
    await query.answer()
    actor = get_or_create_user(update)
    if not await require_name(update, actor):
        return
    if actor["role"] != "admin":
        await query.edit_message_text("Только admin")
        return
    _, action, user_id_s = (query.data or "").split(":")
    user_id = int(user_id_s)
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT name, role FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        await query.edit_message_text("Пользователь не найден.")
        return
    if action == "set_therapist":
        if row["role"] == "admin":
            conn.close()
            await query.edit_message_text("Админа нельзя сделать логопедом")
            return
        cur.execute("UPDATE users SET role = 'therapist' WHERE id = ?", (user_id,))
        conn.commit()
        conn.close()
        await query.edit_message_text(f"Роль логопеда назначена: {row['name'] or user_id}")
        return
    if action == "revoke_therapist":
        if row["role"] != "therapist":
            conn.close()
            await query.edit_message_text("Пользователь не является логопедом.")
            return
        cur.execute("UPDATE users SET role = 'patient' WHERE id = ?", (user_id,))
        conn.commit()
        conn.close()
        await query.edit_message_text(f"Роль логопеда отозвана: {row['name'] or user_id}")
        return
    conn.close()


async def handle_patient_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # загружает и выводит последние попытки выбранного пациента
    query = update.callback_query
    await query.answer()
    therapist = get_or_create_user(update)
    if therapist["role"] not in {"therapist", "admin"}:
        await query.edit_message_text("Команда только для логопеда или админа")
        return
    _, action, patient_id_s = (query.data or "").split(":")
    if action != "results":
        return
    patient_id = int(patient_id_s)
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM patient_therapist_access WHERE patient_id = ? AND therapist_id = ?",
        (patient_id, therapist["id"]),
    )
    if not cur.fetchone():
        conn.close()
        await query.edit_message_text("Нет доступа к этому пациенту")
        return
    cur.execute(
        """
        SELECT a.ml_score, a.created_at, t.text
        FROM attempts a
        JOIN twisters t ON t.id = a.twister_id
        WHERE a.patient_id = ?
        ORDER BY a.id DESC
        LIMIT 10
        """,
        (patient_id,),
    )
    rows = cur.fetchall()
    conn.close()
    if not rows:
        await query.edit_message_text("У пациента пока нет результатов")
        return
    lines = []
    for r in rows:
        verdict = "хорошо" if r["ml_score"] >= 50 else "нарушение"
        date = r["created_at"][:10]
        lines.append(f"{r['ml_score']:.0f}/100 — {verdict} | {r['text'][:30]}... | {date}")
    await query.edit_message_text("Результаты пациента:\n" + "\n".join(lines))


async def get_twister(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # выдаёт скороговорку и сохраняет её id как текущую для пользователя
    user = get_or_create_user(update)
    if not await require_name(update, user):
        return
    letters = json.loads(user["selected_letters"] or "[]")
    twister = pick_twister(letters)
    if not twister:
        await update.message.reply_text("Скороговорки не найдены. Обратитесь к администратору.")
        return
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE users SET current_twister_id = ? WHERE telegram_id = ?",
        (twister["id"], update.effective_user.id),
    )
    conn.commit()
    conn.close()
    await update.message.reply_text(
        f"Ваша скороговорка:\n{twister['text']}\n\nОтправьте голосовое сообщение с произношением."
    )


async def grant_access(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # команда /grant_access: выдаёт логопеду доступ к данным пациента
    patient = get_or_create_user(update)
    if not await require_name(update, patient):
        return
    if len(context.args) == 0:
        await show_grant_access_picker(update.message)
        return
    if len(context.args) != 1 or not context.args[0].isdigit():
        await update.message.reply_text("Использование: /grant_access <telegram_id_логопеда> или кнопка меню")
        return
    therapist_tg = int(context.args[0])
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM users WHERE telegram_id = ? AND role IN ('therapist', 'admin')",
        (therapist_tg,),
    )
    therapist = cur.fetchone()
    if not therapist:
        conn.close()
        await update.message.reply_text("Логопед не найден")
        return
    cur.execute(
        "INSERT OR IGNORE INTO patient_therapist_access (patient_id, therapist_id) VALUES (?, ?)",
        (patient["id"], therapist["id"]),
    )
    conn.commit()
    conn.close()
    await update.message.reply_text("Доступ выдан")


async def revoke_access(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # команда /revoke_access: отзывает доступ логопеда
    patient = get_or_create_user(update)
    if not await require_name(update, patient):
        return
    if len(context.args) == 0:
        await show_revoke_access_picker(update.message, patient["id"])
        return
    if len(context.args) != 1 or not context.args[0].isdigit():
        await update.message.reply_text("Использование: /revoke_access <telegram_id_логопеда> или кнопка меню")
        return
    therapist_tg = int(context.args[0])
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id FROM users WHERE telegram_id = ? AND role IN ('therapist', 'admin')",
        (therapist_tg,),
    )
    row = cur.fetchone()
    if row:
        cur.execute(
            "DELETE FROM patient_therapist_access WHERE patient_id = ? AND therapist_id = ?",
            (patient["id"], row["id"]),
        )
        conn.commit()
    conn.close()
    await update.message.reply_text("Доступ отозван")


async def my_results(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # выводит последние 5 попыток пациента
    user = get_or_create_user(update)
    if not await require_name(update, user):
        return
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT a.id, a.ml_score, a.created_at, t.text
        FROM attempts a
        JOIN twisters t ON t.id = a.twister_id
        WHERE a.patient_id = ?
        ORDER BY a.id DESC
        LIMIT 5
        """,
        (user["id"],),
    )
    rows = cur.fetchall()
    conn.close()
    if not rows:
        await update.message.reply_text("Результатов пока нет")
        return
    lines = []
    for r in rows:
        verdict = "хорошо" if r["ml_score"] >= 50 else "нарушение"
        date = r["created_at"][:10]
        lines.append(f"{r['ml_score']:.0f}/100 — {verdict} | {r['text'][:30]}... | {date}")
    await update.message.reply_text("Последние результаты:\n" + "\n".join(lines))


async def patients(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # выводит список пациентов, выдавших доступ этому логопеду
    therapist = get_or_create_user(update)
    if not await require_name(update, therapist):
        return
    if therapist["role"] not in {"therapist", "admin"}:
        await update.message.reply_text("Команда только для логопеда или админа")
        return
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT u.telegram_id, u.name
        FROM patient_therapist_access pta
        JOIN users u ON u.id = pta.patient_id
        WHERE pta.therapist_id = ?
        """,
        (therapist["id"],),
    )
    rows = cur.fetchall()
    conn.close()
    if not rows:
        await update.message.reply_text("Нет пациентов с доступом")
        return
    await update.message.reply_text(
        "Пациенты:\n" + "\n".join([f"{r['name']} ({r['telegram_id']})" for r in rows])
    )


async def patient_results(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # команда /patient_results: выводит попытки пациента по telegram_id
    therapist = get_or_create_user(update)
    if not await require_name(update, therapist):
        return
    if therapist["role"] not in {"therapist", "admin"}:
        await update.message.reply_text("Команда только для логопеда или админа")
        return
    if len(context.args) != 1 or not context.args[0].isdigit():
        await update.message.reply_text("Использование: /patient_results <telegram_id_пациента>")
        return
    patient_tg = int(context.args[0])
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE telegram_id = ?", (patient_tg,))
    patient = cur.fetchone()
    if not patient:
        conn.close()
        await update.message.reply_text("Пациент не найден")
        return
    cur.execute(
        "SELECT 1 FROM patient_therapist_access WHERE patient_id = ? AND therapist_id = ?",
        (patient["id"], therapist["id"]),
    )
    if not cur.fetchone():
        conn.close()
        await update.message.reply_text("Нет доступа к этому пациенту")
        return
    cur.execute(
        """
        SELECT a.id, a.ml_score, a.created_at, t.text
        FROM attempts a
        JOIN twisters t ON t.id = a.twister_id
        WHERE a.patient_id = ?
        ORDER BY a.id DESC
        LIMIT 10
        """,
        (patient["id"],),
    )
    rows = cur.fetchall()
    conn.close()
    if not rows:
        await update.message.reply_text("У пациента пока нет результатов")
        return
    lines = []
    for r in rows:
        verdict = "хорошо" if r["ml_score"] >= 50 else "нарушение"
        date = r["created_at"][:10]
        lines.append(f"{r['ml_score']:.0f}/100 — {verdict} | {r['text'][:30]}... | {date}")
    await update.message.reply_text("Результаты пациента:\n" + "\n".join(lines))


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # принимает голосовое, конвертирует в WAV, отправляет в ML API, сохраняет попытку
    user = get_or_create_user(update)
    if not await require_name(update, user):
        return
    twister_id = user["current_twister_id"]
    if not twister_id:
        await update.message.reply_text("Сначала вызовите /get_twister")
        return
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT text, has_l, has_r, has_s, has_t, has_c, has_ch, has_sh, has_sch FROM twisters WHERE id = ?",
        (twister_id,),
    )
    tw = cur.fetchone()
    if not tw:
        conn.close()
        await update.message.reply_text("Скороговорка не найдена. Повторите /get_twister")
        return
    letter_flags = json.dumps(
        [tw["has_l"], tw["has_r"], tw["has_s"], tw["has_t"], tw["has_c"], tw["has_ch"], tw["has_sh"], tw["has_sch"]]
    )
    try:
        voice = update.message.voice
        file = await context.bot.get_file(voice.file_id)
        audio_bytes = await file.download_as_bytearray()
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        audio_path = AUDIO_DIR / f"{user['id']}_{twister_id}_{ts}.wav"
        wav_result = subprocess.run(
            ["ffmpeg", "-i", "pipe:0", "-f", "wav", "-ar", "16000", "-ac", "1", "pipe:1", "-loglevel", "quiet"],
            input=bytes(audio_bytes), capture_output=True,
        )
        wav_bytes = wav_result.stdout if wav_result.stdout else bytes(audio_bytes)
        audio_path.write_bytes(wav_bytes)
        files = {"audio": ("voice.wav", wav_bytes, "audio/wav")}
        data = {
            "twister_id": str(twister_id),
            "letters": letter_flags,
            "duration": str(voice.duration),
            "n_speakers": "1",
        }
        r = requests.post(ML_API_URL, files=files, data=data, timeout=30)
        r.raise_for_status()
        ml = r.json()
    except Exception as e:
        conn.close()
        await update.message.reply_text(f"Ошибка ML API: {e}")
        return
    cur.execute(
        """
        INSERT INTO attempts (patient_id, twister_id, selected_letters, audio_path, ml_score, ml_payload, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user["id"],
            twister_id,
            user["selected_letters"] or "[]",
            str(audio_path),
            float(ml.get("score", 0)),
            json.dumps(ml, ensure_ascii=False),
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    conn.close()
    score = ml.get("score", 0)
    verdict = "Нарушение обнаружено" if ml.get("label") == "bad" else "Хорошее произношение"
    await update.message.reply_text(
        f"Результат: {verdict}\n"
        f"Оценка: {score} / 100"
    )


def main():
    # регистрирует все обработчики и запускает long-polling
    if not BOT_TOKEN:
        raise RuntimeError("Set BOT_TOKEN")
    init_db()
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("set_name", set_name))
    app.add_handler(CommandHandler("set_role", set_role))
    app.add_handler(CommandHandler("set_letters", set_letters))
    app.add_handler(CommandHandler("get_twister", get_twister))
    app.add_handler(CommandHandler("grant_access", grant_access))
    app.add_handler(CommandHandler("revoke_access", revoke_access))
    app.add_handler(CommandHandler("my_results", my_results))
    app.add_handler(CommandHandler("patients", patients))
    app.add_handler(CommandHandler("patient_results", patient_results))
    app.add_handler(CallbackQueryHandler(handle_letters_callback, pattern=r"^letters:"))
    app.add_handler(CallbackQueryHandler(handle_access_callback, pattern=r"^access:"))
    app.add_handler(CallbackQueryHandler(handle_role_callback, pattern=r"^role:"))
    app.add_handler(CallbackQueryHandler(handle_patient_callback, pattern=r"^patient:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_menu_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.run_polling()


if __name__ == "__main__":
    main()
