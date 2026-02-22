# 아키텍처 및 시스템 설계도

## 기술 스택 및 환경
- Backend: Python 3.11+, SQLAlchemy 2.x, psycopg2-binary, Alembic
- Web Frontend: **Streamlit**
- Automation: **GitHub Actions (Cron Scheduler)**
- Database: 클라우드 PostgreSQL (Neon 등)
- `.env` 설정 항목: `POSTGRES_DSN`, `GEMINI_API_KEY`, `DISCORD_WEBHOOK_URL`
- 작업 디렉토리: `src/` (백엔드), `web/` (프론트엔드), `.github/workflows/` (클라우드 자동화)

## 모듈 분리 원칙
- `src/data_engine/`: 데이터 수집, QC, 피처 계산 (DB 적재 담당)
- `src/ai_engine/`: LLM 프롬프팅으로 리포트 생성 및 디스코드 전송
- **`web/app.py`: Streamlit으로 DB 데이터를 읽어와 웹 화면에 렌더링 (Read-Only)**

## 데이터베이스 구조
1. **스키마 구성:** `meta`, `raw`, `feat`, `label`, `exp`, `report`
2. **타입 규칙:** 가격/지표는 `DOUBLE PRECISION`, 메타데이터는 `JSONB`

## CLI 커맨드 및 자동화 디자인
- `init-db`: DB 스키마 초기화
- `pipeline-run`: 수집 -> 피처 가공 -> AI 리포트 생성 -> 디스코드 전송을 한 번에 실행하는 통합 커맨드 (GitHub Actions가 매일 이 커맨드를 실행함)