@.ai/1_planning.md 와 @.ai/2_architecture.md 의 설계도를 완벽히 숙지해 주세요.

[Part A]와 [Part B]가 완벽하게 구현되어 DB 적재부터 AI 리포트 생성, Discord 전송까지 모두 성공했습니다.
이번 턴에서는 대망의 최종 단계인 [Part C] 'Web Dashboard 및 Cloud Automation'을 구현합니다.

1. `web/app.py`: Streamlit을 활용한 웹 대시보드 코드를 작성하세요.
   - DB(`POSTGRES_DSN`)에 읽기 전용으로 접속합니다. (database.py 모듈 재사용 권장)
   - 화면 구성 요소: 
     A. **시스템 상태 (Health Check):** `meta.job_run` 테이블을 조회하여 최근 수집 성공 여부 및 날짜 표시
     B. **오늘의 AI 리포트:** `report.daily_reports` 테이블을 조회하여 가장 최근에 생성된 마크다운 리포트 출력
     C. **주요 지표 테이블:** `feat` 스키마에서 최근 영업일 기준 주요 종목들의 현재가, disparity25, rsi14 등을 DataFrame으로 예쁘게 렌더링

2. `.github/workflows/auto_pipeline.yml`: GitHub Actions 자동화 워크플로우 파일을 작성하세요.
   - 매주 화~토요일 오전 7시 (한국 시간 기준, 장 마감 후)에 동작하도록 cron을 설정하세요.
   - Checkout -> Python 3.11 세팅 -> 의존성 설치 -> 환경변수 주입(GitHub Secrets 활용) -> `ingest`, `process`, `ai-report` CLI 커맨드를 순차적으로 실행하는 Step을 구성하세요.

코드 블록 맨 첫 줄에 반드시 생성/수정할 파일 경로를 주석으로 명시하세요.