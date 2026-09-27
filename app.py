import os
import re
import json
import urllib.request
from datetime import date, datetime
from typing import List, Optional
import streamlit as st
import pandas as pd
from pydantic import BaseModel, Field
import httpx
from openai import OpenAI
from dotenv import load_dotenv

# PostgreSQL ORM 라이브러리
from sqlalchemy import create_engine, text

# PDF 생성을 위한 ReportLab 라이브러리
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# .env 환경변수 로드
load_dotenv()

# 고정 참석자 멤버 풀
FIXED_MEMBER_POOL = ["오호민", "신가을", "이준희", "김영석", "송지섭"]


# =========================================================
# 1. Pydantic 스키마 정의
# =========================================================
class PreviousContext(BaseModel):
    past_decisions_summary: Optional[str] = Field(default=None, description="이전 회의 주요 결정 사항")
    action_item_updates: List[str] = Field(default_factory=list, description="이전 액션 아이템 이행 상태")


class SpeakerOpinion(BaseModel):
    speaker_name: str = Field(description="참석자 이름")
    core_stance: str = Field(description="해당 화자의 핵심 관점 및 입장")
    key_arguments: List[str] = Field(description="주요 제안 및 반박 요점 리스트")


class AgendaTopic(BaseModel):
    topic_name: str = Field(description="안건 명")
    background: str = Field(description="논의 배경 요약")
    decisions: List[str] = Field(description="합의된 최종 결정 사항")
    open_issues: List[str] = Field(description="미결 사항")


class ActionItem(BaseModel):
    status: str = Field(default="❌ 미진행", description="진행 상태 ('❌ 미진행', '⏳ 진행중', '✅ 완료' 중 하나)")
    task: str = Field(description="실행 과제 내용")
    owner: str = Field(description="담당자 이름 (언급 없을 시 '미지정')")
    due_date: Optional[date] = Field(default=None, description="마감 기한 (YYYY-MM-DD)")
    memo: str = Field(default="", description="전달사항 또는 진행 메모")


class NextMeetingPlan(BaseModel):
    scheduled_date: Optional[date] = Field(default=None, description="다음 회의 예정 일자 (YYYY-MM-DD)")
    upcoming_agendas: List[str] = Field(default_factory=list, description="다음 회의 주요 안건")


class StructuredMeetingNote(BaseModel):
    meeting_title: str = Field(description="회의 제목")
    meeting_date: date = Field(description="회의 일자 (YYYY-MM-DD)")
    participants: List[str] = Field(description="참석자 명단")
    meeting_objective: str = Field(description="회의 목적")
    previous_context: Optional[PreviousContext] = Field(default=None)
    speaker_opinions: List[SpeakerOpinion] = Field(default_factory=list)
    agenda_topics: List[AgendaTopic] = Field(default_factory=list)
    action_items: List[ActionItem] = Field(default_factory=list)
    next_meeting: Optional[NextMeetingPlan] = Field(default=None)


# =========================================================
# 2. Supabase (PostgreSQL) 데이터베이스 연동 레이어
# =========================================================
@st.cache_resource
def get_db_engine():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        return None
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+psycopg://", 1)
    elif db_url.startswith("postgresql://") and "+psycopg" not in db_url and "+psycopg2" not in db_url:
        db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return create_engine(db_url, pool_pre_ping=True, pool_size=5, max_overflow=10)


def init_db():
    """상시 과제 전용 테이블 자동 생성"""
    engine = get_db_engine()
    if not engine:
        return
    with engine.connect() as conn:
        conn.execute(text("""
                          CREATE TABLE IF NOT EXISTS standalone_tasks
                          (
                              id
                              BIGSERIAL
                              PRIMARY
                              KEY,
                              task
                              TEXT
                              NOT
                              NULL,
                              owner
                              VARCHAR
                          (
                              100
                          ) NOT NULL,
                              due_date DATE,
                              status VARCHAR
                          (
                              50
                          ) DEFAULT '❌ 미진행',
                              memo TEXT DEFAULT '',
                              created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
                              );
                          """))
        conn.commit()


@st.cache_data(ttl=60)
def get_all_meetings_from_supabase() -> List[dict]:
    """저장된 전체 회의록 목록 조회 (최신 일자순)"""
    engine = get_db_engine()
    if not engine:
        return []
    with engine.connect() as conn:
        stmt = text("""
                    SELECT id,
                           title,
                           meeting_date,
                           participants,
                           raw_turns,
                           structured_json,
                           markdown_text,
                           created_at
                    FROM meeting_records
                    ORDER BY meeting_date DESC, id DESC;
                    """)
        result = conn.execute(stmt)
        return [dict(row._mapping) for row in result]


@st.cache_data(ttl=60)
def get_all_standalone_tasks_from_supabase() -> List[dict]:
    """등록된 상시 과제 목록 조회"""
    engine = get_db_engine()
    if not engine:
        return []
    with engine.connect() as conn:
        stmt = text("""
                    SELECT id, task, owner, due_date, status, memo, created_at
                    FROM standalone_tasks
                    ORDER BY due_date ASC NULLS LAST, id DESC;
                    """)
        result = conn.execute(stmt)
        return [dict(row._mapping) for row in result]


def add_standalone_task_to_supabase(task: str, owner: str, due_date_val: Optional[date], status: str, memo: str):
    """상시 과제 신규 저장"""
    engine = get_db_engine()
    if not engine:
        return
    with engine.connect() as conn:
        stmt = text("""
                    INSERT INTO standalone_tasks (task, owner, due_date, status, memo)
                    VALUES (:task, :owner, :due_date, :status, :memo);
                    """)
        conn.execute(stmt, {
            "task": task,
            "owner": owner,
            "due_date": due_date_val,
            "status": status,
            "memo": memo
        })
        conn.commit()
    st.cache_data.clear()


def delete_standalone_task_from_supabase(task_id: int):
    """상시 과제 단건 삭제"""
    engine = get_db_engine()
    if not engine:
        return
    with engine.connect() as conn:
        stmt = text("DELETE FROM standalone_tasks WHERE id = :id;")
        conn.execute(stmt, {"id": task_id})
        conn.commit()
    st.cache_data.clear()


def save_meeting_to_supabase(
        title: str,
        meeting_date: str,
        participants: List[str],
        raw_turns: List[dict],
        structured_note: StructuredMeetingNote,
        markdown_text: str
) -> Optional[int]:
    """Supabase PostgreSQL에 회의록 신규 저장"""
    engine = get_db_engine()
    if not engine:
        raise ValueError("DATABASE_URL이 설정되지 않았습니다.")

    with engine.connect() as conn:
        stmt = text("""
                    INSERT INTO meeting_records (title, meeting_date, participants, raw_turns, structured_json,
                                                 markdown_text)
                    VALUES (:title, :meeting_date, :participants, :raw_turns, :structured_json,
                            :markdown_text) RETURNING id;
                    """)
        result = conn.execute(stmt, {
            "title": title,
            "meeting_date": meeting_date,
            "participants": json.dumps(participants, ensure_ascii=False),
            "raw_turns": json.dumps(raw_turns, ensure_ascii=False),
            "structured_json": structured_note.model_dump_json(),
            "markdown_text": markdown_text
        })
        conn.commit()
        st.cache_data.clear()
        return result.scalar()


def update_meeting_markdown(meeting_id: int, new_markdown: str):
    """편집된 마크다운 텍스트를 DB에 갱신 저장"""
    engine = get_db_engine()
    if not engine:
        return
    with engine.connect() as conn:
        stmt = text("UPDATE meeting_records SET markdown_text = :md WHERE id = :id;")
        conn.execute(stmt, {"md": new_markdown, "id": meeting_id})
        conn.commit()
    st.cache_data.clear()


def update_action_items_unified(meeting_targets: List[dict], standalone_targets: List[dict]):
    """회의 과제와 상시 과제를 각각 원자적으로 안전 갱신"""
    engine = get_db_engine()
    if not engine:
        return

    with engine.connect() as conn:
        if meeting_targets:
            grouped = {}
            for item in meeting_targets:
                grouped.setdefault(item["meeting_id"], []).append(item)

            for m_id, items in grouped.items():
                stmt = text("SELECT structured_json FROM meeting_records WHERE id = :id;")
                row = conn.execute(stmt, {"id": m_id}).fetchone()
                if not row:
                    continue

                raw_val = row[0]
                s_data = json.loads(raw_val) if isinstance(raw_val, str) else raw_val
                action_items = s_data.get("action_items", [])

                for it in items:
                    idx = it["item_idx"]
                    if idx < len(action_items):
                        action_items[idx]["status"] = it["new_status"]
                        action_items[idx]["memo"] = it["new_memo"]
                        action_items[idx]["due_date"] = it["new_due_date"]

                s_data["action_items"] = action_items

                try:
                    note_obj = StructuredMeetingNote.model_validate(s_data)
                    new_md = render_to_markdown(note_obj)
                except Exception:
                    new_md = None

                if new_md:
                    upd_stmt = text("""
                                    UPDATE meeting_records
                                    SET structured_json = :sj,
                                        markdown_text   = :md
                                    WHERE id = :id;
                                    """)
                    conn.execute(upd_stmt, {
                        "sj": json.dumps(s_data, ensure_ascii=False),
                        "md": new_md,
                        "id": m_id
                    })
                else:
                    upd_stmt = text("UPDATE meeting_records SET structured_json = :sj WHERE id = :id;")
                    conn.execute(upd_stmt, {
                        "sj": json.dumps(s_data, ensure_ascii=False),
                        "id": m_id
                    })

        if standalone_targets:
            for st_item in standalone_targets:
                upd_st = text("""
                              UPDATE standalone_tasks
                              SET status   = :status,
                                  memo     = :memo,
                                  due_date = :due_date
                              WHERE id = :id;
                              """)
                conn.execute(upd_st, {
                    "id": st_item["task_id"],
                    "status": st_item["new_status"],
                    "memo": st_item["new_memo"],
                    "due_date": st_item["new_due_date"]
                })

        conn.commit()
    st.cache_data.clear()


def delete_meeting_from_supabase(meeting_id: int):
    """특정 회의록 DB 영구 삭제"""
    engine = get_db_engine()
    if not engine:
        return
    with engine.connect() as conn:
        stmt = text("DELETE FROM meeting_records WHERE id = :id;")
        conn.execute(stmt, {"id": meeting_id})
        conn.commit()
    st.cache_data.clear()


def get_meeting_by_id_from_supabase(meeting_id: int) -> Optional[dict]:
    """특정 회의록 단건 상세 조회"""
    engine = get_db_engine()
    if not engine:
        return None
    with engine.connect() as conn:
        stmt = text("SELECT * FROM meeting_records WHERE id = :id LIMIT 1;")
        result = conn.execute(stmt, {"id": meeting_id}).fetchone()
        return dict(result._mapping) if result else None


def format_past_meeting_as_context(structured_json_str: str) -> str:
    """과거 회의 JSON에서 다음 회의 프롬프트에 주입할 컨텍스트 추출"""
    try:
        data = json.loads(structured_json_str) if isinstance(structured_json_str, str) else structured_json_str
        decisions_list = []
        for topic in data.get("agenda_topics", []):
            for d in topic.get("decisions", []):
                decisions_list.append(f"[{topic.get('topic_name')}] {d}")

        action_list = []
        for act in data.get("action_items", []):
            st_mark = act.get("status", "❌ 미진행")
            due_str = f" (~{act.get('due_date')})" if act.get('due_date') else ""
            memo_str = f" (메모: {act.get('memo')})" if act.get('memo') else ""
            action_list.append(f"{st_mark} {act.get('task')} (담당: {act.get('owner')}{due_str}){memo_str}")

        open_issues = []
        for topic in data.get("agenda_topics", []):
            for o in topic.get("open_issues", []):
                open_issues.append(f"[{topic.get('topic_name')}] {o}")

        context_lines = []
        if decisions_list:
            context_lines.append("[지난 회의 주요 결정 사항]\n" + "\n".join(f"- {d}" for d in decisions_list))
        if action_list:
            context_lines.append("[지난 회의 과제 현황 및 진행 상태]\n" + "\n".join(f"- {a}" for a in action_list))
        if open_issues:
            context_lines.append("[지난 회의 미결 과제]\n" + "\n".join(f"- {o}" for o in open_issues))

        return "\n\n".join(context_lines)
    except Exception:
        return "이전 회의 데이터 파싱 실패"


# =========================================================
# 3. 텍스트 파싱 유틸리티
# =========================================================
def parse_raw_text_to_turns(raw_text: str) -> List[dict]:
    turns = []
    lines = raw_text.strip().split("\n")
    pattern = re.compile(r"^([^:\n\r]+?)\s*:\s*(.*)$")

    current_speaker = None
    current_content = []

    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue
        match = pattern.match(line_str)
        if match:
            if current_speaker and current_content:
                turns.append({"speaker": current_speaker, "content": " ".join(current_content)})
            current_speaker = match.group(1).strip()
            current_content = [match.group(2).strip()]
        else:
            if current_speaker:
                current_content.append(line_str)
            else:
                current_speaker = "참석자"
                current_content = [line_str]

    if current_speaker and current_content:
        turns.append({"speaker": current_speaker, "content": " ".join(current_content)})
    return turns


# =========================================================
# 4. 마크다운 변환 렌더러
# =========================================================
def render_to_markdown(note: StructuredMeetingNote) -> str:
    lines = []
    lines.append(f"# {note.meeting_title} 회의록\n")
    lines.append("---\n")
    lines.append("## 1. 회의 개요")
    lines.append("| 항목 | 내용 |")
    lines.append("| :--- | :--- |")
    lines.append(f"| **회의 일자** | {note.meeting_date.strftime('%Y-%m-%d')} |")
    lines.append(f"| **참석자** | {', '.join(note.participants) if note.participants else '미지정'} |")
    lines.append(f"| **회의 목적** | {note.meeting_objective.replace(chr(10), ' ')} |\n")
    lines.append("---\n")

    lines.append("## 2. 이전 회의 팔로업 (Previous Context)")
    if note.previous_context and (
            note.previous_context.past_decisions_summary or note.previous_context.action_item_updates):
        past = note.previous_context.past_decisions_summary or "기록된 이전 결정 사항 없음"
        lines.append(f"- **지난 회의 주요 결정 사항**: {past}")
        lines.append("- **지난 액션 아이템 진행 상태**:")
        if note.previous_context.action_item_updates:
            for upd in note.previous_context.action_item_updates:
                lines.append(f"  - {upd}")
        else:
            lines.append("  - 이전 진행 상태 업데이트 없음")
    else:
        lines.append("- *이전 회의 기록 또는 팔로업 사항이 없습니다.*")
    lines.append("\n---\n")

    lines.append("## 3. 화자별 핵심 의견 종합 (Speaker Opinions)")
    lines.append("> 각 참석자가 회의 전반에 걸쳐 개진한 주요 관점, 제안, 우려 사항을 종합합니다.\n")
    if note.speaker_opinions:
        for op in note.speaker_opinions:
            lines.append(f"- **[{op.speaker_name}]**:")
            lines.append(f"  - **주요 관점**: {op.core_stance}")
            if op.key_arguments:
                lines.append("  - **핵심 발언/제안**:")
                for arg in op.key_arguments:
                    lines.append(f"    - {arg}")
    else:
        lines.append("- *기록된 화자별 의견이 없습니다.*")
    lines.append("\n---\n")

    lines.append("## 4. 안건별 논의 및 결과 (Decisions & Open Issues)\n")
    if note.agenda_topics:
        for idx, ag in enumerate(note.agenda_topics, 1):
            lines.append(f"### 안건 {idx}. {ag.topic_name}")
            lines.append(f"* **논의 배경**: {ag.background}")
            lines.append("* **결정된 내용 (Decisions)**:")
            if ag.decisions:
                for d in ag.decisions:
                    lines.append(f"  - {d}")
            else:
                lines.append("  - *결정된 사항 없음*")
            lines.append("* **미결 내용 (Open Issues)**:")
            if ag.open_issues:
                for o in ag.open_issues:
                    lines.append(f"  - {o}")
            else:
                lines.append("  - *미결 사항 없음*")
            lines.append("")
    else:
        lines.append("- *등록된 세부 안건이 없습니다.*\n")
    lines.append("---\n")

    lines.append("## 5. 실행 과제 (Action Items)")
    if note.action_items:
        lines.append("| 상태 | 번호 | 작업 내용 (Task) | 담당자 (Owner) | 마감 기한 (Due Date) | 메모 (Memo) |")
        lines.append("| :---: | :---: | :--- | :---: | :---: | :--- |")
        for idx, item in enumerate(note.action_items, 1):
            due = item.due_date.strftime("%Y-%m-%d") if item.due_date else "미지정"
            s_task = item.task.replace("\n", " ").replace("|", "\\|")
            s_memo = (item.memo or "-").replace("\n", " ").replace("|", "\\|")
            lines.append(f"| {item.status} | {idx} | {s_task} | {item.owner} | {due} | {s_memo} |")
    else:
        lines.append("- *도출된 액션 아이템이 없습니다.*")
    lines.append("\n---\n")

    lines.append("## 6. 차기 회의 계획 (Next Meeting)")
    if note.next_meeting:
        sc = note.next_meeting.scheduled_date.strftime("%Y-%m-%d") if note.next_meeting.scheduled_date else "추후 협의"
        lines.append(f"- **차기 회의 예정 일자**: {sc}")
        lines.append("- **다룰 주요 안건**:")
        if note.next_meeting.upcoming_agendas:
            for u in note.next_meeting.upcoming_agendas:
                lines.append(f"  - {u}")
        else:
            lines.append("  - *예정된 안건 없음*")
    else:
        lines.append("- *차기 회의 계획이 설정되지 않았습니다.*")

    return "\n".join(lines)


# =========================================================
# 5. 한글 PDF 생성 엔진
# =========================================================
@st.cache_resource
def load_korean_font_cached() -> str:
    font_name = "NanumGothic"
    local_font_path = "NanumGothic.ttf"
    if font_name in pdfmetrics.getRegisteredFontNames():
        return font_name

    if not os.path.exists(local_font_path):
        font_url = "https://raw.githubusercontent.com/google/fonts/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
        try:
            urllib.request.urlretrieve(font_url, local_font_path)
        except Exception:
            win_font = "C:/Windows/Fonts/malgun.ttf"
            if os.path.exists(win_font):
                pdfmetrics.registerFont(TTFont(font_name, win_font))
                return font_name
            return "Helvetica"

    try:
        pdfmetrics.registerFont(TTFont(font_name, local_font_path))
        return font_name
    except Exception:
        return "Helvetica"


def generate_pdf_bytes(note: StructuredMeetingNote) -> bytes:
    from io import BytesIO
    buffer = BytesIO()
    font_name = load_korean_font_cached()

    doc = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=36, bottomMargin=36
    )
    elements = []

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontName=font_name, fontSize=18, leading=22,
                                 textColor=colors.HexColor("#1A202C"))
    h2_style = ParagraphStyle('SectionH2', parent=styles['Heading2'], fontName=font_name, fontSize=12, leading=16,
                              textColor=colors.HexColor("#2B6CB0"), spaceBefore=10, spaceAfter=6)
    body_style = ParagraphStyle('BodyKR', parent=styles['Normal'], fontName=font_name, fontSize=8.5, leading=12,
                                textColor=colors.HexColor("#2D3748"))
    bold_style = ParagraphStyle('BoldKR', parent=body_style, fontName=font_name, textColor=colors.HexColor("#1A202C"))

    elements.append(Paragraph(f"{note.meeting_title} 회의록", title_style))
    elements.append(Spacer(1, 10))

    overview_data = [
        [Paragraph("<b>회의 일자</b>", bold_style), Paragraph(note.meeting_date.strftime("%Y-%m-%d"), body_style)],
        [Paragraph("<b>참석자</b>", bold_style), Paragraph(", ".join(note.participants), body_style)],
        [Paragraph("<b>회의 목적</b>", bold_style), Paragraph(note.meeting_objective, body_style)]
    ]
    t_overview = Table(overview_data, colWidths=[80, 455])
    t_overview.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(t_overview)
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("화자별 핵심 의견 종합", h2_style))
    for op in note.speaker_opinions:
        elements.append(Paragraph(f"<b>[{op.speaker_name}]</b> - {op.core_stance}", bold_style))
        for arg in op.key_arguments:
            elements.append(Paragraph(f"• {arg}", body_style))
        elements.append(Spacer(1, 3))

    elements.append(Paragraph("안건별 논의 및 결정 사항", h2_style))
    for idx, ag in enumerate(note.agenda_topics, 1):
        elements.append(Paragraph(f"<b>안건 {idx}. {ag.topic_name}</b>", bold_style))
        elements.append(Paragraph(f"<b>배경</b>: {ag.background}", body_style))
        if ag.decisions:
            elements.append(Paragraph("<b>[결정 사항]</b> " + ", ".join(ag.decisions), body_style))
        if ag.open_issues:
            elements.append(Paragraph("<b>[미결 과제]</b> " + ", ".join(ag.open_issues), body_style))
        elements.append(Spacer(1, 4))

    elements.append(Paragraph("실행 과제 (Action Items)", h2_style))
    if note.action_items:
        act_data = [[
            Paragraph("<b>상태</b>", bold_style),
            Paragraph("<b>No</b>", bold_style),
            Paragraph("<b>작업 내용</b>", bold_style),
            Paragraph("<b>담당자</b>", bold_style),
            Paragraph("<b>마감일</b>", bold_style),
            Paragraph("<b>메모</b>", bold_style)
        ]]
        for idx, item in enumerate(note.action_items, 1):
            due = item.due_date.strftime("%Y-%m-%d") if item.due_date else "미지정"
            act_data.append([
                Paragraph(item.status, body_style),
                Paragraph(str(idx), body_style),
                Paragraph(item.task, body_style),
                Paragraph(item.owner, body_style),
                Paragraph(due, body_style),
                Paragraph(item.memo or "-", body_style)
            ])
        t_act = Table(act_data, colWidths=[55, 25, 190, 60, 65, 140])
        t_act.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(t_act)

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


# =========================================================
# 6. 비밀번호 게이트 인증 함수 (Option A)
# =========================================================
def check_password() -> bool:
    """팀 비밀번호 인증 게이트 (올바른 비밀번호 입력 전까지 전체 UI 차단)"""
    if st.session_state.get("authenticated", False):
        return True

    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([3, 4, 3])
    with c2:
        st.markdown("<h2 style='text-align: center;'>🔒 팀 내부 시스템 접근 인증</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: gray;'>인가된 팀원 전용 시스템입니다. 비밀번호를 입력해주세요.</p>",
                    unsafe_allow_html=True)

        with st.form("team_login_form", clear_on_submit=False):
            input_pw = st.text_input("비밀번호 (TEAM_PASSWORD)", type="password", placeholder="비밀번호 입력")
            submit = st.form_submit_button("인증 및 접속하기", use_container_width=True, type="primary")

            if submit:
                # 1. Streamlit Secrets 우선 확인, 2. OS 환경변수 확인, 3. 기본값 'team2' 확인
                target_pw = "team2"
                if hasattr(st, "secrets") and "TEAM_PASSWORD" in st.secrets:
                    target_pw = str(st.secrets["TEAM_PASSWORD"]).strip()
                elif os.getenv("TEAM_PASSWORD"):
                    target_pw = os.getenv("TEAM_PASSWORD").strip()

                if input_pw.strip() == target_pw:
                    st.session_state["authenticated"] = True
                    st.success("✅ 인증 성공! 시스템을 로드합니다.")
                    st.rerun()
                else:
                    st.error("❌ 비밀번호가 올바르지 않습니다.")

    return False


# =========================================================
# 7. Streamlit 메인 애플리케이션
# =========================================================
def main():
    st.set_page_config(page_title="팀 회의록 & 과제 관리 시스템", layout="wide", page_icon="📝")

    # [보안 게이트] 비밀번호 인증이 완료되지 않으면 실행 중단
    if not check_password():
        st.stop()

    api_key = os.getenv("OPENAI_API_KEY")
    db_url = os.getenv("DATABASE_URL")

    # DB 테이블 자동 점검 및 생성
    if db_url:
        try:
            init_db()
        except Exception:
            pass

    # DB에 저장된 회의록 목록 선행 조회
    all_existing_meetings = get_all_meetings_from_supabase()

    # 세션 상태 초기화: 저장된 회의가 있다면 가장 최신 회의 맥락을 기본값으로 자동 로드
    if "prev_context_buffer" not in st.session_state:
        if all_existing_meetings:
            st.session_state.prev_context_buffer = format_past_meeting_as_context(
                all_existing_meetings[0]['structured_json'])
        else:
            st.session_state.prev_context_buffer = ""

    if "turns" not in st.session_state:
        st.session_state.turns = []
    if "generated_note" not in st.session_state:
        st.session_state.generated_note = None
    if "markdown_output" not in st.session_state:
        st.session_state.markdown_output = None
    if "current_record_id" not in st.session_state:
        st.session_state.current_record_id = None
    if "action_status_card_filter" not in st.session_state:
        st.session_state.action_status_card_filter = "전체"

    # 사이드바
    with st.sidebar:
        st.header("⚙️ 시스템 설정")
        if api_key and api_key.strip():
            st.success("✅ OpenAI Key 정상")
        else:
            st.error("❌ `OPENAI_API_KEY` 없음")

        if db_url and db_url.strip():
            st.success("✅ Supabase DB 정상")
        else:
            st.error("❌ `DATABASE_URL` 없음")

        model_name = st.selectbox("LLM 모델", ["gpt-4o", "gpt-4o-mini"], index=0)

        st.markdown("---")
        st.subheader("👥 팀 기본 멤버 (5인)")
        for member in FIXED_MEMBER_POOL:
            st.markdown(f"- **{member}**")

        st.markdown("---")
        if st.button("🔒 로그아웃", use_container_width=True):
            st.session_state["authenticated"] = False
            st.rerun()

    # 4대 탭 구성
    tab_new, tab_history, tab_actions, tab_standalone = st.tabs([
        "📝 새 회의 작성 및 정리",
        "☁️ Supabase 회의록 보관함",
        "📌 팀 과제(Action Items) 현황판",
        "➕ 상시 과제 직접 등록"
    ])

    # -----------------------------------------------------
    # TAB 1: 새 회의 작성 및 정리 (직전 회의 맥락 자동 연동)
    # -----------------------------------------------------
    with tab_new:
        left_col, right_col = st.columns([5, 5])

        with left_col:
            st.subheader("📋 회의 기본 정보")
            m_title = st.text_input("회의 제목", value="주간 개발 및 프로젝트 동기화 회의")

            c_date, c_part = st.columns([4, 6])
            with c_date:
                m_date = st.date_input("회의 일자", value=date.today())
            with c_part:
                selected_participants = st.multiselect(
                    "금일 참석자 (불참자 제외)",
                    options=FIXED_MEMBER_POOL,
                    default=FIXED_MEMBER_POOL
                )

            st.markdown("##### 🔗 이전 회의 팔로업 연동")
            if all_existing_meetings:
                ctx_choices = {}
                for m in all_existing_meetings:
                    is_latest = (m['id'] == all_existing_meetings[0]['id'])
                    tag = " (최근 회의 / 자동 선택됨)" if is_latest else ""
                    ctx_choices[f"[{m['meeting_date']}] {m['title']}{tag}"] = m['id']
                ctx_choices["연동 안 함 (새 프로젝트 회의)"] = None

                c_sel, c_btn = st.columns([7, 3])
                with c_sel:
                    chosen_ctx_label = st.selectbox(
                        "연동할 이전 회의 선택",
                        options=list(ctx_choices.keys()),
                        index=0,
                        help="선택한 회의의 결정 사항과 미결 과제가 아래 입력창에 즉시 주입됩니다."
                    )
                with c_btn:
                    st.write("")
                    st.write("")
                    if st.button("🔄 맥락 다시 불러오기", use_container_width=True):
                        target_id = ctx_choices[chosen_ctx_label]
                        if target_id:
                            m_target = get_meeting_by_id_from_supabase(target_id)
                            if m_target:
                                st.session_state.prev_context_buffer = format_past_meeting_as_context(
                                    m_target['structured_json'])
                                st.success("선택한 회의 맥락이 반영되었습니다!")
                                st.rerun()
                        else:
                            st.session_state.prev_context_buffer = ""
                            st.info("이전 맥락을 비웠습니다.")
                            st.rerun()
            else:
                st.caption("ℹ️ DB에 등록된 이전 회의가 없습니다. (이번 회의가 첫 번째 회의로 기록됩니다)")

            prev_context_text = st.text_area(
                "이전 회의 맥락 / 팔로업 (직전 회의 데이터가 자동 반영되어 있습니다)",
                value=st.session_state.prev_context_buffer,
                height=110,
                placeholder="직전 회의 결정 사항 및 액션 아이템 내용이 자동으로 표시됩니다."
            )
            st.session_state.prev_context_buffer = prev_context_text

            st.markdown("---")
            st.subheader("💬 대화 내용 입력")

            input_tab1, input_tab2 = st.tabs(["⚡ 한 줄씩 실시간 입력", "📋 외부 대화 일괄 붙여넣기 (이름:대화)"])

            with input_tab1:
                if not selected_participants:
                    st.warning("금일 참석자를 1명 이상 선택해주세요.")
                else:
                    with st.form("turn_input_form", clear_on_submit=True):
                        c_spk, c_cnt = st.columns([3, 7])
                        with c_spk:
                            selected_speaker = st.selectbox("화자 선택", options=selected_participants)
                        with c_cnt:
                            turn_text = st.text_input("발언 내용", placeholder="예: 2안은 유지보수 비용이 이중으로 발생합니다.")

                        if st.form_submit_button("턴 추가 (+)", use_container_width=True) and turn_text.strip():
                            st.session_state.turns.append({
                                "speaker": selected_speaker,
                                "content": turn_text.strip()
                            })
                            st.rerun()

            with input_tab2:
                st.caption("메신저에서 '이름: 내용' 형식으로 복사한 전체 대화를 붙여넣으세요.")
                bulk_text = st.text_area(
                    "외부 텍스트 붙여넣기",
                    height=120,
                    placeholder="오호민: 이번 주 배포 일정 확인 부탁드립니다.\n신가을: QA 테스트 완료되어 내일 배포 가능합니다."
                )
                b_c1, b_c2 = st.columns(2)
                with b_c1:
                    if st.button("📥 로그에 추가하기", use_container_width=True):
                        if bulk_text.strip():
                            parsed_turns = parse_raw_text_to_turns(bulk_text)
                            st.session_state.turns.extend(parsed_turns)
                            st.success(f"총 {len(parsed_turns)}개의 발언이 추가되었습니다!")
                            st.rerun()
                        else:
                            st.warning("텍스트를 입력해주세요.")
                with b_c2:
                    if st.button("🔄 기존 로그 비우고 교체", use_container_width=True):
                        if bulk_text.strip():
                            parsed_turns = parse_raw_text_to_turns(bulk_text)
                            st.session_state.turns = parsed_turns
                            st.success(f"{len(parsed_turns)}개의 발언으로 교체되었습니다!")
                            st.rerun()
                        else:
                            st.warning("텍스트를 입력해주세요.")

            st.markdown("##### 📜 실시간 누적 로그")
            if not st.session_state.turns:
                st.info("아직 입력된 발언이 없습니다.")
            else:
                c_clear, _ = st.columns([3, 7])
                if c_clear.button("전체 로그 초기화", use_container_width=True):
                    st.session_state.turns = []
                    st.rerun()

                log_box = st.container(height=240)
                for idx, turn in enumerate(st.session_state.turns):
                    with log_box:
                        r1, r2 = st.columns([9, 1])
                        r1.markdown(f"**{turn['speaker']}**: {turn['content']}")
                        if r2.button("🗑️", key=f"del_turn_{idx}"):
                            st.session_state.turns.pop(idx)
                            st.rerun()

            st.markdown("---")
            if st.button("🚀 전체 저장 후 회의록 정리", type="primary", use_container_width=True):
                if not api_key or not db_url:
                    st.error("API 키 또는 DB URL 설정을 확인하세요.")
                elif not selected_participants or not st.session_state.turns:
                    st.warning("참석자 및 발언 내용을 1개 이상 입력하세요.")
                else:
                    with st.spinner("LLM 분석 및 Supabase 클라우드 저장 중..."):
                        try:
                            custom_http_client = httpx.Client(headers={"Accept-Encoding": "gzip, deflate"})
                            client = OpenAI(api_key=api_key, http_client=custom_http_client)

                            turns_log = "\n".join([f"{t['speaker']}: {t['content']}" for t in st.session_state.turns])
                            system_prompt = (
                                "당신은 대화 로그를 분석하여 공식 회의록을 작성하는 전문 비즈니스 AI입니다.\n"
                                "1. [이전 회의 맥락]이 제공된 경우, 이전 회의의 결정 사항과 과제가 이번 회의에서 어떻게 다루어졌는지 파악하여 "
                                "   [previous_context.past_decisions_summary]와 [previous_context.action_item_updates]에 구체적으로 정리하십시오.\n"
                                "2. 발언의 문맥을 분석하여 [화자별 핵심 의견], [안건 논의 배경]을 정밀 도출하십시오.\n"
                                "3. 합의된 사항은 [결정 사항], 이견이 남거나 보류된 사항은 [미결 과제]로 구분하십시오.\n"
                                "4. 액션 아이템(action_items) 도출 시:\n"
                                "   - status는 기본값인 '❌ 미진행'으로 설정하십시오.\n"
                                "   - memo는 대화 중 해당 과제와 관련해 특별히 언급된 유의사항이나 참고사항이 있다면 작성하고, 없으면 빈 문자열('')로 두십시오.\n"
                                "5. 일자는 반드시 YYYY-MM-DD 형식만 추출하며, 대화에 없는 내용은 절대 지어내지 마십시오.\n\n"
                                f"반드시 아래 JSON 스키마를 엄격히 준수하여 응답하십시오:\n{json.dumps(StructuredMeetingNote.model_json_schema(), ensure_ascii=False)}"
                            )
                            user_prompt = (
                                f"[회의 개요]\n- 제목: {m_title}\n- 일자: {m_date.strftime('%Y-%m-%d')}\n"
                                f"- 참석자: {', '.join(selected_participants)}\n"
                                f"- 이전 회의 맥락 (주입됨):\n{st.session_state.prev_context_buffer or '없음'}\n\n"
                                f"[순수 발언 로그]:\n{turns_log}"
                            )

                            response = client.chat.completions.create(
                                model=model_name,
                                messages=[{"role": "system", "content": system_prompt},
                                          {"role": "user", "content": user_prompt}],
                                response_format={"type": "json_object"},
                                temperature=0.1
                            )

                            parsed = StructuredMeetingNote.model_validate_json(response.choices[0].message.content)
                            md_output = render_to_markdown(parsed)

                            rec_id = save_meeting_to_supabase(
                                title=parsed.meeting_title,
                                meeting_date=parsed.meeting_date.strftime('%Y-%m-%d'),
                                participants=parsed.participants,
                                raw_turns=st.session_state.turns,
                                structured_note=parsed,
                                markdown_text=md_output
                            )

                            st.session_state.generated_note = parsed
                            st.session_state.markdown_output = md_output
                            st.session_state.current_record_id = rec_id
                            st.success(f"회의록 정리 완료 및 DB 저장 성공! (기록 ID: {rec_id})")
                            st.rerun()
                        except Exception as e:
                            st.error(f"생성 중 오류 발생: {e}")

        with right_col:
            st.subheader("📄 생성된 회의록")
            if st.session_state.markdown_output and st.session_state.generated_note:
                col_d1, col_d2 = st.columns(2)
                col_d1.download_button(
                    "📥 마크다운 다운로드",
                    data=st.session_state.markdown_output,
                    file_name=f"{m_date.strftime('%Y%m%d')}_{m_title}_회의록.md",
                    mime="text/markdown",
                    use_container_width=True
                )
                try:
                    pdf_bytes = generate_pdf_bytes(st.session_state.generated_note)
                    col_d2.download_button(
                        "📥 PDF 다운로드",
                        data=pdf_bytes,
                        file_name=f"{m_date.strftime('%Y%m%d')}_{m_title}_회의록.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as p_err:
                    col_d2.error(f"PDF 생성 오류: {p_err}")

                with st.expander("📋 메신저 공유용 마크다운 복사하기"):
                    st.caption("우측 상단 복사 아이콘을 누르면 클립보드에 복사됩니다.")
                    st.code(st.session_state.markdown_output, language="markdown")

                edit_mode = st.toggle("✏️ 마크다운 직접 수정 모드", key="toggle_edit_new")
                if edit_mode:
                    edited_md = st.text_area("마크다운 내용 편집", value=st.session_state.markdown_output, height=500)
                    if st.button("💾 수정한 내용 Supabase DB에 갱신", type="primary"):
                        st.session_state.markdown_output = edited_md
                        if st.session_state.current_record_id:
                            update_meeting_markdown(st.session_state.current_record_id, edited_md)
                            st.success("수정 사항이 Supabase에 안전하게 반영되었습니다!")
                            st.rerun()
                else:
                    with st.container(height=550):
                        st.markdown(st.session_state.markdown_output)
            else:
                st.info("좌측에서 대화를 입력하고 정리 버튼을 누르면 회의록이 표시됩니다.")

    # -----------------------------------------------------
    # TAB 2: Supabase 회의록 보관함
    # -----------------------------------------------------
    with tab_history:
        st.subheader("☁️ Supabase 팀 회의록 보관함")
        all_meetings = get_all_meetings_from_supabase()

        if not all_meetings:
            st.info("저장된 회의록이 없습니다. 새 회의를 먼저 작성해 보세요.")
        else:
            with st.expander("🔍 회의록 검색 및 필터링 옵션", expanded=True):
                f_c1, f_c2, f_c3 = st.columns([4, 3, 3])
                with f_c1:
                    search_kw = st.text_input("키워드 검색 (제목, 안건, 결정사항, 본문)", placeholder="예: 인덱스, 배포, Supabase")
                with f_c2:
                    filter_member = st.multiselect("특정 참석자 포함 필터", options=FIXED_MEMBER_POOL)
                with f_c3:
                    st.caption("회의 일자 필터링")
                    use_date_filter = st.checkbox("일자 범위 지정")
                    if use_date_filter:
                        d_range = st.date_input("기간 선택", value=(date(2026, 1, 1), date.today()))
                    else:
                        d_range = None

            filtered = []
            for m in all_meetings:
                m_full_text = f"{m['title']} {m['markdown_text']} {m.get('participants', '')}".lower()
                if search_kw.strip() and search_kw.strip().lower() not in m_full_text:
                    continue
                if filter_member:
                    m_parts = json.loads(m['participants']) if isinstance(m['participants'], str) else m['participants']
                    if not any(mem in m_parts for mem in filter_member):
                        continue
                if use_date_filter and isinstance(d_range, tuple) and len(d_range) == 2:
                    try:
                        m_dt = datetime.strptime(m['meeting_date'], "%Y-%m-%d").date()
                        if not (d_range[0] <= m_dt <= d_range[1]):
                            continue
                    except Exception:
                        pass
                filtered.append(m)

            st.caption(f"검색 결과: 총 {len(filtered)}건 / 전체 {len(all_meetings)}건")

            if not filtered:
                st.warning("조건에 부합하는 회의록이 없습니다.")
            else:
                col_list, col_view = st.columns([4, 6])

                with col_list:
                    options = {
                        f"[{m['meeting_date']}] {m['title']} (ID: {m['id']})": m['id']
                        for m in filtered
                    }
                    sel_label = st.selectbox("조회할 회의 선택", list(options.keys()))
                    sel_id = options[sel_label]
                    detail = get_meeting_by_id_from_supabase(sel_id)

                    if detail:
                        st.markdown("---")
                        if st.button("📌 이 회의를 '새 회의의 이전 맥락'으로 불러오기", use_container_width=True):
                            extracted_ctx = format_past_meeting_as_context(detail['structured_json'])
                            st.session_state.prev_context_buffer = extracted_ctx
                            st.success("맥락 복사 완료! [새 회의 작성] 탭에서 확인하세요.")

                        st.markdown("---")
                        st.markdown("##### ⚠️ 회의록 관리")
                        with st.expander("🗑️ 이 회의록 삭제하기"):
                            st.warning("삭제 시 Supabase에서 영구 제거됩니다.")
                            confirm_del = st.checkbox("정말로 이 회의록을 삭제하시겠습니까?", key=f"chk_del_{detail['id']}")
                            if st.button("삭제 실행", type="primary", disabled=not confirm_del,
                                         key=f"btn_del_{detail['id']}"):
                                delete_meeting_from_supabase(detail['id'])
                                st.success("회의록이 삭제되었습니다.")
                                st.rerun()

                        with st.expander("🔍 원본 발언 로그 보기"):
                            raw_data = detail['raw_turns'] if isinstance(detail['raw_turns'], list) else json.loads(
                                detail['raw_turns'])
                            for r in raw_data:
                                st.write(f"- **{r['speaker']}**: {r['content']}")

                with col_view:
                    if detail:
                        st.markdown(f"#### 📖 {detail['title']} (일자: {detail['meeting_date']})")

                        d_c1, d_c2 = st.columns(2)
                        d_c1.download_button(
                            "📥 마크다운 다운로드",
                            data=detail['markdown_text'],
                            file_name=f"{detail['meeting_date']}_{detail['title']}.md",
                            mime="text/markdown",
                            key=f"hist_md_btn_{detail['id']}",
                            use_container_width=True
                        )
                        try:
                            note_dict = detail['structured_json'] if isinstance(detail['structured_json'],
                                                                                dict) else json.loads(
                                detail['structured_json'])
                            note_obj = StructuredMeetingNote.model_validate(note_dict)
                            pdf_data = generate_pdf_bytes(note_obj)
                            d_c2.download_button(
                                "📥 PDF 다운로드",
                                data=pdf_data,
                                file_name=f"{detail['meeting_date']}_{detail['title']}.pdf",
                                mime="application/pdf",
                                key=f"hist_pdf_btn_{detail['id']}",
                                use_container_width=True
                            )
                        except Exception as h_pe:
                            d_c2.caption(f"PDF 생성 불가: {h_pe}")

                        with st.expander("📋 메신저 공유용 마크다운 복사하기"):
                            st.code(detail['markdown_text'], language="markdown")

                        edit_hist_mode = st.toggle("✏️ 마크다운 편집 모드", key=f"edit_hist_mode_{detail['id']}")
                        if edit_hist_mode:
                            hist_edited_text = st.text_area("마크다운 내용 편집", value=detail['markdown_text'], height=500,
                                                            key=f"area_edit_{detail['id']}")
                            if st.button("💾 수정한 내용 DB에 즉시 갱신", type="primary", key=f"save_edit_{detail['id']}"):
                                update_meeting_markdown(detail['id'], hist_edited_text)
                                st.success("수정 사항이 Supabase에 업데이트되었습니다!")
                                st.rerun()
                        else:
                            st.markdown("---")
                            with st.container(height=550):
                                st.markdown(detail['markdown_text'])

    # -----------------------------------------------------
    # TAB 3: 팀 과제(Action Items) 통합 현황판
    # -----------------------------------------------------
    with tab_actions:
        st.subheader("📌 팀 과제(Action Items) 통합 현황판")
        st.caption("회의록 과제와 상시 과제를 통합 관리합니다. 표에서 수정한 뒤 '💾 저장'을 누르면 DB에 안전하게 반영됩니다.")

        all_records = get_all_meetings_from_supabase()
        standalone_records = get_all_standalone_tasks_from_supabase()
        aggregated_items = []

        for rec in all_records:
            try:
                s_data = rec['structured_json'] if isinstance(rec['structured_json'], dict) else json.loads(
                    rec['structured_json'])
                for idx, item in enumerate(s_data.get("action_items", [])):
                    st_val = item.get("status") or "❌ 미진행"
                    if st_val not in ["❌ 미진행", "⏳ 진행중", "✅ 완료"]:
                        st_val = "❌ 미진행"

                    raw_due = item.get("due_date")
                    due_date_obj = None
                    if raw_due:
                        try:
                            if isinstance(raw_due, (date, datetime)):
                                due_date_obj = raw_due if isinstance(raw_due, date) else raw_due.date()
                            else:
                                due_date_obj = datetime.strptime(str(raw_due)[:10], "%Y-%m-%d").date()
                        except Exception:
                            due_date_obj = None

                    aggregated_items.append({
                        "_source": "meeting",
                        "_meeting_id": rec['id'],
                        "_item_idx": idx,
                        "_task_id": None,
                        "상태": st_val,
                        "담당자": item.get("owner", "미지정"),
                        "실행 과제 (Task)": item.get("task", ""),
                        "마감 기한": due_date_obj,
                        "메모": item.get("memo", ""),
                        "출처": f"회의: {rec['title']}",
                        "등록/회의 일자": rec['meeting_date']
                    })
            except Exception:
                continue

        for st_row in standalone_records:
            st_val = st_row.get("status") or "❌ 미진행"
            if st_val not in ["❌ 미진행", "⏳ 진행중", "✅ 완료"]:
                st_val = "❌ 미진행"

            due_date_obj = st_row.get("due_date")
            if due_date_obj and isinstance(due_date_obj, str):
                try:
                    due_date_obj = datetime.strptime(due_date_obj[:10], "%Y-%m-%d").date()
                except Exception:
                    due_date_obj = None

            aggregated_items.append({
                "_source": "standalone",
                "_meeting_id": None,
                "_item_idx": None,
                "_task_id": st_row["id"],
                "상태": st_val,
                "담당자": st_row.get("owner", "미지정"),
                "실행 과제 (Task)": st_row.get("task", ""),
                "마감 기한": due_date_obj,
                "메모": st_row.get("memo", ""),
                "출처": "📌 상시 직접 등록",
                "등록/회의 일자": str(st_row["created_at"])[:10]
            })

        if not aggregated_items:
            st.info("등록된 과제(Action Items)가 없습니다.")
        else:
            df_tasks = pd.DataFrame(aggregated_items)

            owner_filter = st.selectbox(
                "👤 담당자 필터링",
                options=["전체 팀원 보기"] + FIXED_MEMBER_POOL + ["미지정"],
                key="filter_owner_actions"
            )
            base_df = df_tasks if owner_filter == "전체 팀원 보기" else df_tasks[df_tasks["담당자"] == owner_filter]

            cnt_total = len(base_df)
            cnt_x = len(base_df[base_df["상태"] == "❌ 미진행"])
            cnt_p = len(base_df[base_df["상태"] == "⏳ 진행중"])
            cnt_d = len(base_df[base_df["상태"] == "✅ 완료"])

            st.write("▼ **상태 카드를 클릭하면 해당 목록만 아래 표에 표시됩니다**")
            b_col1, b_col2, b_col3, b_col4 = st.columns(4)

            curr_status = st.session_state.action_status_card_filter

            with b_col1:
                label_all = f"📁 전체 과제 ({cnt_total}건)" + ("  👈" if curr_status == "전체" else "")
                if st.button(label_all, use_container_width=True,
                             type="primary" if curr_status == "전체" else "secondary"):
                    st.session_state.action_status_card_filter = "전체"
                    st.rerun()

            with b_col2:
                label_x = f"❌ 미진행 ({cnt_x}건)" + ("  👈" if curr_status == "❌ 미진행" else "")
                if st.button(label_x, use_container_width=True,
                             type="primary" if curr_status == "❌ 미진행" else "secondary"):
                    st.session_state.action_status_card_filter = "❌ 미진행"
                    st.rerun()

            with b_col3:
                label_p = f"⏳ 진행중 ({cnt_p}건)" + ("  👈" if curr_status == "⏳ 진행중" else "")
                if st.button(label_p, use_container_width=True,
                             type="primary" if curr_status == "⏳ 진행중" else "secondary"):
                    st.session_state.action_status_card_filter = "⏳ 진행중"
                    st.rerun()

            with b_col4:
                label_d = f"✅ 완료 ({cnt_d}건)" + ("  👈" if curr_status == "✅ 완료" else "")
                if st.button(label_d, use_container_width=True,
                             type="primary" if curr_status == "✅ 완료" else "secondary"):
                    st.session_state.action_status_card_filter = "✅ 완료"
                    st.rerun()

            if st.session_state.action_status_card_filter == "전체":
                display_df = base_df.copy()
            else:
                display_df = base_df[base_df["상태"] == st.session_state.action_status_card_filter].copy()

            st.markdown("---")

            edited_df = st.data_editor(
                display_df,
                column_config={
                    "_source": None,
                    "_meeting_id": None,
                    "_item_idx": None,
                    "_task_id": None,
                    "상태": st.column_config.SelectboxColumn(
                        "상태 (클릭 변경)",
                        options=["❌ 미진행", "⏳ 진행중", "✅ 완료"],
                        required=True,
                        width="small"
                    ),
                    "담당자": st.column_config.TextColumn("담당자", disabled=True, width="small"),
                    "실행 과제 (Task)": st.column_config.TextColumn("실행 과제 (Task)", disabled=True, width="large"),
                    "마감 기한": st.column_config.DateColumn(
                        "마감 기한 (더블클릭 변경)",
                        format="YYYY-MM-DD",
                        width="small"
                    ),
                    "메모": st.column_config.TextColumn("메모 / 코멘트 (더블클릭 작성)", width="large"),
                    "출처": st.column_config.TextColumn("출처", disabled=True),
                    "등록/회의 일자": st.column_config.TextColumn("일자", disabled=True),
                },
                hide_index=True,
                use_container_width=True,
                key="action_items_interactive_editor"
            )

            if st.button("💾 상태·기한·메모 변경사항 DB에 안전 저장", type="primary", use_container_width=True):
                modified_meeting_targets = []
                modified_standalone_targets = []

                orig_lookup = {
                    (item["_source"], item["_meeting_id"], item["_item_idx"], item["_task_id"]): item
                    for item in aggregated_items
                }

                def to_iso_date(val):
                    if pd.isna(val) or val is None:
                        return None
                    if isinstance(val, (datetime, pd.Timestamp)):
                        return val.strftime("%Y-%m-%d")
                    if isinstance(val, date):
                        return val.strftime("%Y-%m-%d")
                    s = str(val).strip()
                    return s if s else None

                for _, row in edited_df.iterrows():
                    key = (row["_source"], row["_meeting_id"], row["_item_idx"], row["_task_id"])
                    if key in orig_lookup:
                        orig = orig_lookup[key]
                        new_st = row["상태"]
                        new_mem = str(row["메모"]).strip() if pd.notna(row["메모"]) else ""
                        new_due_str = to_iso_date(row["마감 기한"])
                        orig_due_str = to_iso_date(orig["마감 기한"])

                        if (new_st != orig["상태"]) or (new_mem != orig["메모"]) or (new_due_str != orig_due_str):
                            if row["_source"] == "meeting":
                                modified_meeting_targets.append({
                                    "meeting_id": row["_meeting_id"],
                                    "item_idx": int(row["_item_idx"]),
                                    "new_status": new_st,
                                    "new_memo": new_mem,
                                    "new_due_date": new_due_str
                                })
                            else:
                                modified_standalone_targets.append({
                                    "task_id": row["_task_id"],
                                    "new_status": new_st,
                                    "new_memo": new_mem,
                                    "new_due_date": new_due_str
                                })

                if not modified_meeting_targets and not modified_standalone_targets:
                    st.info("변경된 내용이 없습니다.")
                else:
                    update_action_items_unified(modified_meeting_targets, modified_standalone_targets)
                    total_mod = len(modified_meeting_targets) + len(modified_standalone_targets)
                    st.success(f"총 {total_mod}건의 과제가 DB에 안전하게 반영되었습니다!")
                    st.rerun()

    # -----------------------------------------------------
    # TAB 4: 상시 과제 직접 등록
    # -----------------------------------------------------
    with tab_standalone:
        st.subheader("➕ 상시 과제 직접 등록")
        st.caption("정규 회의록 외에 일상 업무, 긴급 버그 수정 등을 등록하여 팀 현황판에 공유합니다.")

        c_form, c_view = st.columns([5, 5])

        with c_form:
            with st.form("new_standalone_task_form", clear_on_submit=True):
                st.markdown("##### 📝 과제 정보 입력")
                st_task_name = st.text_input("과제 내용 (Task)", placeholder="예: Supabase 백업 정책 수립 및 자동화")

                f_c1, f_c2 = st.columns(2)
                with f_c1:
                    st_owner = st.selectbox("담당자 지정", options=FIXED_MEMBER_POOL + ["미지정"])
                with f_c2:
                    st_status = st.selectbox("초기 상태", options=["❌ 미진행", "⏳ 진행중", "✅ 완료"], index=0)

                st_due_date = st.date_input("마감 기한 (선택 사항)", value=date.today())
                st_memo = st.text_area("메모 및 참고사항", placeholder="예: AWS S3 버킷 설정 필요", height=80)

                submitted = st.form_submit_button("과제 등록하기 (+)", type="primary", use_container_width=True)
                if submitted:
                    if not st_task_name.strip():
                        st.warning("과제 내용을 입력해주세요.")
                    else:
                        add_standalone_task_to_supabase(
                            task=st_task_name.strip(),
                            owner=st_owner,
                            due_date_val=st_due_date,
                            status=st_status,
                            memo=st_memo.strip()
                        )
                        st.success("상시 과제가 등록되었습니다! [📌 팀 과제 현황판]에서도 확인할 수 있습니다.")
                        st.rerun()

        with c_view:
            st.markdown("##### 📋 등록된 상시 과제 관리")
            current_standalone = get_all_standalone_tasks_from_supabase()

            if not current_standalone:
                st.info("등록된 상시 과제가 없습니다.")
            else:
                for t in current_standalone:
                    with st.container(border=True):
                        t_c1, t_c2 = st.columns([8, 2])
                        with t_c1:
                            st.markdown(f"**{t['status']} {t['task']}**")
                            d_str = t['due_date'].strftime('%Y-%m-%d') if t.get('due_date') else '기한 없음'
                            st.caption(f"👤 담당: **{t['owner']}** | 📅 마감: {d_str}")
                            if t.get('memo'):
                                st.caption(f"💬 메모: {t['memo']}")
                        with t_c2:
                            if st.button("삭제", key=f"del_st_task_{t['id']}", use_container_width=True):
                                delete_standalone_task_from_supabase(t['id'])
                                st.success("과제가 삭제되었습니다.")
                                st.rerun()


if __name__ == "__main__":
    main()
