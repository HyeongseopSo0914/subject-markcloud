"""
간단한 상표 검색 API 구현
제공된 trademark_sample.json 데이터를 사용하세요.
데이터 내 다양한 결측치(null 값), 리스트 형태의 필드가 존재합니다. 이를 적절히 처리해주세요.

"""

"""
productName: 상표명(한글)
productNameEng: 상표명(영문)
applicationNumber: 출원 번호
applicationDate: 출원일 (YYYYMMDD 형식)
registerStatus: 상표 등록 상태 (등록, 실효, 거절, 출원 등)
publicationNumber: 공고 번호
publicationDate: 공고일 (YYYYMMDD 형식)
registrationNumber: 등록 번호
registrationDate: 등록일 (YYYYMMDD 형식)
internationalRegNumbers: 국제 출원 번호
internationalRegDate: 국제출원일 (YYYYMMDD 형식)
priorityClaimNumList: 우선권 번호
priorityClaimDateList: 우선권 일자 (YYYYMMDD 형식)
asignProductMainCodeList: 상품 주 분류 코드 리스트
asignProductSubCodeList: 상품 유사군 코드 리스트
viennaCodeList: 비엔나 코드 리스트

"""
"""

필수 요구사항
상표 데이터를 검색할 수 있는 API를 1개 이상 구현하세요.
Python 기반의 FastAPI 프레임워크를 사용해 개발해 주세요.
최소한 하나 이상의 필터링 기능을 포함하세요. (예: 등록 상태, 상품 코드, 날짜 등)
구현에 대한 상세한 조건은 따로 제시하지 않습니다. 본인이 자유롭게 설계하세요.

"""
"""
GitHub 레파지토리에 소스코드를 올려주세요.
GitHub 주소와 함께 소스코드를 압축하여 jmhwang@markcloud.co.kr 메일로 보내주세요
README.md 파일에는 다음 내용을 꼭 포함해주세요.
API 사용법 및 실행 방법
구현된 기능 설명
기술적 의사결정에 대한 설명
문제 해결 과정에서 고민했던 점
개선하고 싶은 부분 (선택사항)
"""
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import uvicorn
import json

app = FastAPI()

# 데이터 로드
with open("trademark_final_sample_fixed_cleaned.json", encoding="utf-8") as f:
    data = json.load(f)

@app.get("/search")
def search_trademark(q: str = Query(..., description="검색어 (productName 또는 applicationNumber)")):
    q = q.strip()
    if not q:
        return JSONResponse(status_code=400, content={"error": "검색어를 입력하세요. ?q=검색어"})

    results = [
        item for item in data
        if q in item.get("productName", "") or q in item.get("applicationNumber", "")
    ]

    if not results:
        return JSONResponse(status_code=404, content={"message": "검색 결과가 없습니다."})

    return results
