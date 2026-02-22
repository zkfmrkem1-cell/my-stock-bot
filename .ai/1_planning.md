# 주식 매매 AI 학습용 DB 생성 및 웹 리포팅 엔진 기획서

## 목표
[Data Engine]으로 주식 데이터를 가공하고, [AI Report Engine]을 통해 전체 요약 리포트를 생성한다. 이 모든 과정은 **클라우드상에서 매일 완전 자동 구동**되며, 최종 결과물(DB 상태 및 AI 리포트)은 **사용자 전용 웹 대시보드**를 통해 한눈에 확인 가능하도록 구축한다.

## 마일스톤 (Four-Track System)

### [Part A: Data Engine (DB 구축 및 데이터 파이프라인)]
- STEP 1: DB 스캐폴딩 및 프로덕션 스키마 (meta, raw, feat, label 스키마 구축)
- STEP 2: Raw 데이터 증분 수집 (yfinance) 및 무결성 검증 (QC)
- STEP 3: Feature 계산 (이격도, RSI 등) 및 Label 생성

### [Part B: AI Report Engine (분석 및 리포팅)]
- STEP 4: 최신 가공 데이터를 바탕으로 AI에게 전체 시황 및 과매도 종목 요약 리포트 작성 요청
- STEP 5: 생성된 AI 리포트와 주요 지표를 DB(`report` 스키마)에 적재하고 디스코드로 전송

### [Part C: Web Dashboard (시각화 및 확인)]
- STEP 6: **Streamlit 기반 웹 대시보드 구축** (DB 적재 상태, 최근 추출된 엑셀 데이터, AI 리포트를 웹페이지에서 한 번에 조회)

### [Part C: Cloud Automation (무인 자동화)]
- STEP 7: **GitHub Actions 기반 스케줄러 세팅** (매일 장 마감 후 파이프라인 자동 실행) 및 Streamlit Cloud를 통한 대시보드 무료 배포