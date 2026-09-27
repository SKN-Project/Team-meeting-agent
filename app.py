import os
import re
import json
import urllib.request
from datetime import date
from typing import List, Optional
import streamlit as st
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
    task: str = Field(description="실행 과제 내용")
    owner: str = Field(description="담당자 이름 (언급 없을 시 '미지정')")
    due_date: Optional[date] = Field(default=None, description="마감 기한 (YYYY-MM-DD)")


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
def get_db_engine():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        return None
    # SQLAlchemy 드라이버 URI 보정
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+psycopg://", 1)
    elif db_url.startswith("postgresql://") and "+psycopg" not in db_url and "+psycopg2" not in db_url:
        db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return create_engine(db_url, pool_pre_ping=True)


def save_meeting_to_supabase(
        title: str,
        meeting_date: str,
        participants: List[str],
        raw_turns: List[dict],
        structured_note: StructuredMeetingNote,
        markdown_text: str
) -> Optional[int]:
    """Supabase PostgreSQL에 회의록 저장 후 ID 반환"""
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
        return result.scalar()


def get_all_meetings_from_supabase() -> List[dict]:
    """저장된 전체 회의록 목록 조회 (최신순)"""
    engine = get_db_engine()
    if not engine:
        return []
    with engine.connect() as conn:
        stmt = text("""
                    SELECT id, title, meeting_date, participants, created_at
                    FROM meeting_records
                    ORDER BY meeting_date DESC, id DESC;
                    """)
        result = conn.execute(stmt)
        return [dict(row._mapping) for row in result]


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
            action_list.append(f"{act.get('task')} (담당: {act.get('owner')})")

        open_issues = []
        for topic in data.get("agenda_topics", []):
            for o in topic.get("open_issues", []):
                open_issues.append(f"[{topic.get('topic_name')}] {o}")

        context_lines = []
        if decisions_list:
            context_lines.append("[지난 회의 주요 결정 사항]\n" + "\n".join(f"- {d}" for d in decisions_list))
        if action_list:
            context_lines.append("[지난 회의 과제 현황]\n" + "\n".join(f"- {a}" for a in action_list))
        if open_issues:
            context_lines.append("[지난 회의 미결 과제]\n" + "\n".join(f"- {o}" for o in open_issues))

        return "\n\n".join(context_lines)
    except Exception:
        return "이전 회의 데이터 파싱 실패"


# =========================================================
# 3. 텍스트 일괄 파싱 유틸리티 함수
# =========================================================
def parse_raw_text_to_turns(raw_text: str) -> List[dict]:
    """
    '이름: 대화' 또는 '이름 : 대화' 형태의 멀티라인 텍스트를 파싱.
    대화 내용 중간의 줄바꿈도 이전 발언에 자동으로 이어붙임.
    """
    turns = []
    lines = raw_text.strip().split("\n")
    # 이름과 발언을 구분하는 정규식: 시작 부분의 '이름: 내용' 패턴 매칭
    pattern = re.compile(r"^([^:\n\r]+?)\s*:\s*(.*)$")

    current_speaker = None
    current_content = []

    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue

        match = pattern.match(line_str)
        if match:
            # 이전 발언 저장
            if current_speaker and current_content:
                turns.append({
                    "speaker": current_speaker,
                    "content": " ".join(current_content)
                })
            current_speaker = match.group(1).strip()
            current_content = [match.group(2).strip()]
        else:
            # 콜론(:)이 없는 줄은 이전 화자의 발언이 이어진 것으로 처리
            if current_speaker:
                current_content.append(line_str)
            else:
                # 첫 줄부터 이름이 없는 경우 기본 화자 지정
                current_speaker = "참석자"
                current_content = [line_str]

    if current_speaker and current_content:
        turns.append({
            "speaker": current_speaker,
            "content": " ".join(current_content)
        })

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
    if note.previous_context:
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
        lines.append("| 번호 | 작업 내용 (Task) | 담당자 (Owner) | 마감 기한 (Due Date) |")
        lines.append("| :---: | :--- | :---: | :---: |")
        for idx, item in enumerate(note.action_items, 1):
            due = item.due_date.strftime("%Y-%m-%d") if item.due_date else "미지정"
            s_task = item.task.replace("\n", " ").replace("|", "\\|")
            lines.append(f"| {idx} | {s_task} | {item.owner} | {due} |")
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
# 5. 한글 PDF 생성 엔진 (CDN 한글 폰트 자동 탑재)
# =========================================================
def get_korean_font_name() -> str:
    """Linux 배포 환경 및 로컬 환경에서 네모(■) 깨짐을 방지하는 폰트 로더"""
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
    font_name = get_korean_font_name()

    doc = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=40, bottomMargin=40
    )
    elements = []

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Heading1'], fontName=font_name, fontSize=18, leading=22,
        textColor=colors.HexColor("#1A202C")
    )
    h2_style = ParagraphStyle(
        'SectionH2', parent=styles['Heading2'], fontName=font_name, fontSize=12, leading=16,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=10, spaceAfter=6
    )
    body_style = ParagraphStyle(
        'BodyKR', parent=styles['Normal'], fontName=font_name, fontSize=9, leading=13,
        textColor=colors.HexColor("#2D3748")
    )
    bold_style = ParagraphStyle('BoldKR', parent=body_style, fontName=font_name, textColor=colors.HexColor("#1A202C"))

    elements.append(Paragraph(f"{note.meeting_title} 회의록", title_style))
    elements.append(Spacer(1, 10))

    overview_data = [
        [Paragraph("<b>회의 일자</b>", bold_style), Paragraph(note.meeting_date.strftime("%Y-%m-%d"), body_style)],
        [Paragraph("<b>참석자</b>", bold_style), Paragraph(", ".join(note.participants), body_style)],
        [Paragraph("<b>회의 목적</b>", bold_style), Paragraph(note.meeting_objective, body_style)]
    ]
    t_overview = Table(overview_data, colWidths=[80, 440])
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
        act_data = [[Paragraph("<b>No</b>", bold_style), Paragraph("<b>작업 내용</b>", bold_style),
                     Paragraph("<b>담당자</b>", bold_style), Paragraph("<b>마감일</b>", bold_style)]]
        for idx, item in enumerate(note.action_items, 1):
            due = item.due_date.strftime("%Y-%m-%d") if item.due_date else "미지정"
            act_data.append(
                [Paragraph(str(idx), body_style), Paragraph(item.task, body_style), Paragraph(item.owner, body_style),
                 Paragraph(due, body_style)])
        t_act = Table(act_data, colWidths=[30, 310, 80, 100])
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
# 6. Streamlit 메인 애플리케이션
# =========================================================
def main():
    st.set_page_config(page_title="팀 공유 회의록 관리 시스템", layout="wide", page_icon="📝")

    api_key = os.getenv("OPENAI_API_KEY")
    db_url = os.getenv("DATABASE_URL")

    # 세션 상태 초기화
    if "turns" not in st.session_state:
        st.session_state.turns = []
    if "generated_note" not in st.session_state:
        st.session_state.generated_note = None
    if "markdown_output" not in st.session_state:
        st.session_state.markdown_output = None
    if "prev_context_buffer" not in st.session_state:
        st.session_state.prev_context_buffer = ""

    # 사이드바
    with st.sidebar:
        st.header("⚙️ 환경 설정")
        if api_key and api_key.strip():
            st.success("✅ OpenAI Key 로드됨")
        else:
            st.error("❌ `OPENAI_API_KEY` 없음")

        if db_url and db_url.strip():
            st.success("✅ Supabase DB 로드됨")
        else:
            st.error("❌ `DATABASE_URL` 없음")

        model_name = st.selectbox("LLM 모델", ["gpt-4o", "gpt-4o-mini"], index=0)

        st.markdown("---")
        st.subheader("👥 팀 기본 멤버 (5인)")
        for member in FIXED_MEMBER_POOL:
            st.markdown(f"- **{member}**")

    # 상단 탭 구성
    tab_new, tab_history = st.tabs(["📝 새 회의 작성 및 정리", "☁️ Supabase 클라우드 보관함"])

    # -----------------------------------------------------
    # TAB 1: 새 회의 작성 및 정리
    # -----------------------------------------------------
    with tab_new:
        left_col, right_col = st.columns([5, 5])

        with left_col:
            st.subheader("📋 회의 기본 정보")
            m_title = st.text_input("회의 제목", value="주간 개발 및 프로젝트 동기화 회의")

            c_date, c_part = st.columns([4, 6])
            with c_date:
                m_date = st.date_input("회의 일자 (시간 제외)", value=date.today())
            with c_part:
                selected_participants = st.multiselect(
                    "금일 참석자 선택 (불참자 제외)",
                    options=FIXED_MEMBER_POOL,
                    default=FIXED_MEMBER_POOL
                )

            prev_context_text = st.text_area(
                "이전 회의 맥락 / 팔로업 (보관함 탭에서 1클릭 복사 가능)",
                value=st.session_state.prev_context_buffer,
                height=80,
                placeholder="지난 회의 결정 사항이나 액션 아이템이 들어갑니다."
            )
            st.session_state.prev_context_buffer = prev_context_text

            st.markdown("---")
            st.subheader("💬 대화 내용 입력")

            # [신규 기능] 탭으로 실시간 1건 입력 vs 외부 텍스트 일괄 붙여넣기 분기
            input_mode_tab1, input_mode_tab2 = st.tabs(["⚡ 한 줄씩 실시간 입력", "📋 외부 대화 일괄 붙여넣기 (이름:대화)"])

            with input_mode_tab1:
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

            with input_mode_tab2:
                st.caption("카카오톡, 슬랙, 회의 메모 등에서 '이름: 내용' 형식으로 복사한 텍스트를 붙여넣으세요.")
                bulk_text = st.text_area(
                    "외부 텍스트 붙여넣기",
                    height=130,
                    placeholder="오호민: 이번 주 배포 일정 확인 부탁드립니다.\n신가을: QA 테스트 완료되어 내일 배포 가능합니다.\n김영석: 서버 인프라 모니터링 준비해두겠습니다."
                )

                col_b1, col_b2 = st.columns([1, 1])
                with col_b1:
                    if st.button("📥 로그에 추가하기", use_container_width=True):
                        if bulk_text.strip():
                            parsed_turns = parse_raw_text_to_turns(bulk_text)
                            st.session_state.turns.extend(parsed_turns)
                            st.success(f"총 {len(parsed_turns)}개의 발언이 로그에 추가되었습니다!")
                            st.rerun()
                        else:
                            st.warning("붙여넣은 텍스트가 없습니다.")
                with col_b2:
                    if st.button("🔄 기존 로그 비우고 덮어쓰기", use_container_width=True):
                        if bulk_text.strip():
                            parsed_turns = parse_raw_text_to_turns(bulk_text)
                            st.session_state.turns = parsed_turns
                            st.success(f"{len(parsed_turns)}개의 발언으로 로그가 교체되었습니다!")
                            st.rerun()
                        else:
                            st.warning("붙여넣은 텍스트가 없습니다.")

            st.markdown("##### 📜 실시간 누적 로그")
            if not st.session_state.turns:
                st.info("아직 입력된 발언이 없습니다.")
            else:
                c_clear, _ = st.columns([3, 7])
                if c_clear.button("전체 로그 초기화", use_container_width=True):
                    st.session_state.turns = []
                    st.rerun()

                log_box = st.container(height=260)
                for idx, turn in enumerate(st.session_state.turns):
                    with log_box:
                        r1, r2 = st.columns([9, 1])
                        r1.markdown(f"**{turn['speaker']}**: {turn['content']}")
                        if r2.button("🗑️", key=f"del_turn_{idx}"):
                            st.session_state.turns.pop(idx)
                            st.rerun()

            st.markdown("---")
            if st.button("🚀 전체 저장 후 회의록 정리", type="primary", use_container_width=True):
                if not api_key:
                    st.error("OpenAI API 키가 설정되지 않았습니다.")
                elif not db_url:
                    st.error("Supabase DATABASE_URL이 설정되지 않았습니다.")
                elif not selected_participants:
                    st.warning("참석자가 선택되지 않았습니다.")
                elif not st.session_state.turns:
                    st.warning("기록된 발언이 없습니다.")
                else:
                    with st.spinner("LLM 분석 및 Supabase 클라우드 저장 중..."):
                        try:
                            custom_http_client = httpx.Client(headers={"Accept-Encoding": "gzip, deflate"})
                            client = OpenAI(api_key=api_key, http_client=custom_http_client)

                            turns_log = "\n".join([f"{t['speaker']}: {t['content']}" for t in st.session_state.turns])

                            system_prompt = (
                                "당신은 순수 대화 로그(Turn-by-turn logs)를 분석하여 공식 회의록을 작성하는 전문 비즈니스 AI입니다.\n\n"
                                "지침:\n"
                                "1. 발언의 내용과 문맥을 읽고 제안, 반박, 동의, 의사결정 여부를 자체 판단하십시오.\n"
                                "2. 화자 간 상호작용 맥락을 추적하여 [화자별 핵심 의견] 및 [안건 논의 배경]에 정밀 반영하십시오.\n"
                                "3. 합의된 사항은 [결정 사항], 이견이 남거나 보류된 사항은 [미결 과제]로 엄격히 분리하십시오.\n"
                                "4. 회의 일자 및 액션 아이템 마감일은 반드시 YYYY-MM-DD 형식의 '날짜만' 추출하십시오.\n"
                                "5. 대화에 없는 내용이나 일정을 임의로 지어내지 마십시오 (Zero Hallucination).\n\n"
                                "반드시 아래 JSON 스키마를 엄격히 준수하여 유효한 JSON 객체만 반환하십시오:\n"
                                f"{json.dumps(StructuredMeetingNote.model_json_schema(), ensure_ascii=False)}"
                            )

                            user_prompt = (
                                f"[회의 개요]\n"
                                f"- 제목: {m_title}\n"
                                f"- 일자: {m_date.strftime('%Y-%m-%d')}\n"
                                f"- 참석자: {', '.join(selected_participants)}\n"
                                f"- 이전 회의 맥락: {st.session_state.prev_context_buffer or '없음'}\n\n"
                                f"[순수 발언 로그]:\n{turns_log}"
                            )

                            response = client.chat.completions.create(
                                model=model_name,
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_prompt}
                                ],
                                response_format={"type": "json_object"},
                                temperature=0.1
                            )

                            raw_json = response.choices[0].message.content
                            parsed = StructuredMeetingNote.model_validate_json(raw_json)
                            md_output = render_to_markdown(parsed)

                            record_id = save_meeting_to_supabase(
                                title=parsed.meeting_title,
                                meeting_date=parsed.meeting_date.strftime('%Y-%m-%d'),
                                participants=parsed.participants,
                                raw_turns=st.session_state.turns,
                                structured_note=parsed,
                                markdown_text=md_output
                            )

                            st.session_state.generated_note = parsed
                            st.session_state.markdown_output = md_output
                            st.success(f"회의록 정리 완료 및 Supabase 저장 성공! (기록 ID: {record_id})")
                            st.rerun()

                        except Exception as e:
                            st.error(f"생성 중 오류 발생: {e}")

        with right_col:
            st.subheader("📄 생성된 회의록")
            if st.session_state.markdown_output and st.session_state.generated_note:
                d_col1, d_col2 = st.columns(2)
                d_col1.download_button(
                    "📥 마크다운(.md) 다운로드",
                    data=st.session_state.markdown_output,
                    file_name=f"{m_date.strftime('%Y%m%d')}_{m_title}_회의록.md",
                    mime="text/markdown",
                    use_container_width=True
                )
                try:
                    pdf_bytes = generate_pdf_bytes(st.session_state.generated_note)
                    d_col2.download_button(
                        "📥 PDF(.pdf) 다운로드",
                        data=pdf_bytes,
                        file_name=f"{m_date.strftime('%Y%m%d')}_{m_title}_회의록.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as p_err:
                    d_col2.error(f"PDF 생성 오류: {p_err}")

                st.markdown("---")
                with st.container(height=650):
                    st.markdown(st.session_state.markdown_output)
            else:
                st.info("좌측에서 발언을 추가하고 [🚀 전체 저장 후 회의록 정리]를 누르면 결과물이 표시됩니다.")

    # -----------------------------------------------------
    # TAB 2: Supabase 클라우드 보관함
    # -----------------------------------------------------
    with tab_history:
        st.subheader("☁️ Supabase 팀 회의록 보관함")

        try:
            meetings = get_all_meetings_from_supabase()
        except Exception as db_err:
            st.error(f"DB 연결 오류: {db_err}")
            meetings = []

        if not meetings:
            st.info("클라우드 DB에 저장된 회의록이 없습니다. 새 회의록을 먼저 작성해보세요.")
        else:
            col_list, col_view = st.columns([4, 6])

            with col_list:
                st.markdown("##### 🗂️ 저장된 회의 목록 (팀원 공용)")
                options = {
                    f"[{m['meeting_date']}] {m['title']} (ID: {m['id']})": m['id']
                    for m in meetings
                }
                selected_label = st.selectbox("조회할 회의 선택", list(options.keys()))
                selected_id = options[selected_label]

                detail = get_meeting_by_id_from_supabase(selected_id)

                if detail:
                    st.markdown("---")
                    st.markdown("##### ⚡ 이전 회의 맥락 자동 연동")
                    st.caption("선택한 회의의 결정 사항과 액션 아이템을 새 회의의 맥락으로 복사합니다.")
                    if st.button("📌 이 회의를 '새 회의의 이전 맥락'으로 불러오기", use_container_width=True):
                        extracted_ctx = format_past_meeting_as_context(detail['structured_json'])
                        st.session_state.prev_context_buffer = extracted_ctx
                        st.success("새 회의 이전 맥락에 복사되었습니다! [새 회의 작성] 탭에서 확인하세요.")

                    st.markdown("---")
                    with st.expander("🔍 원본 발언 로그 보기"):
                        raw_data = detail['raw_turns'] if isinstance(detail['raw_turns'], list) else json.loads(
                            detail['raw_turns'])
                        for r in raw_data:
                            st.write(f"- **{r['speaker']}**: {r['content']}")

            with col_view:
                if detail:
                    st.markdown(f"#### 📖 {detail['title']} (일자: {detail['meeting_date']})")

                    col_hd1, col_hd2 = st.columns(2)
                    col_hd1.download_button(
                        "📥 마크다운 다운로드",
                        data=detail['markdown_text'],
                        file_name=f"{detail['meeting_date']}_{detail['title']}.md",
                        mime="text/markdown",
                        key=f"hist_md_{detail['id']}",
                        use_container_width=True
                    )

                    try:
                        note_dict = detail['structured_json'] if isinstance(detail['structured_json'],
                                                                            dict) else json.loads(
                            detail['structured_json'])
                        note_obj = StructuredMeetingNote.model_validate(note_dict)
                        hist_pdf = generate_pdf_bytes(note_obj)
                        col_hd2.download_button(
                            "📥 PDF 다운로드",
                            data=hist_pdf,
                            file_name=f"{detail['meeting_date']}_{detail['title']}.pdf",
                            mime="application/pdf",
                            key=f"hist_pdf_{detail['id']}",
                            use_container_width=True
                        )
                    except Exception as he:
                        col_hd2.caption(f"PDF 생성 불가: {he}")

                    st.markdown("---")
                    with st.container(height=600):
                        st.markdown(detail['markdown_text'])


if __name__ == "__main__":
    main()