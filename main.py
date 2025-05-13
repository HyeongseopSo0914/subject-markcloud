"""
목표
간단한 상표 검색 API 구현
제공된 trademark_sample.json 데이터를 사용하세요.
데이터 내 다양한 결측치(null 값), 리스트 형태의 필드가 존재합니다. 이를 적절히 처리해주세요.
-------------------------------------------------------------------------------------
필수 요구사항
상표 데이터를 검색할 수 있는 API를 1개 이상 구현하세요.
Python 기반의 FastAPI 프레임워크를 사용해 개발해 주세요.
최소한 하나 이상의 필터링 기능을 포함하세요. (예: 등록 상태, 상품 코드, 날짜 등)
구현에 대한 상세한 조건은 따로 제시하지 않습니다. 본인이 자유롭게 설계하세요.
-------------------------------------------------------------------------------------
GitHub 레파지토리에 소스코드를 올려주세요.
GitHub 주소와 함께 소스코드를 압축하여 jmhwang@markcloud.co.kr 메일로 보내주세요
README.md 파일에는 다음 내용을 꼭 포함해주세요.
API 사용법 및 실행 방법
구현된 기능 설명
기술적 의사결정에 대한 설명
문제 해결 과정에서 고민했던 점
개선하고 싶은 부분 (선택사항)
"""
import warnings
warnings.filterwarnings("ignore")
import uvicorn
import json
import pandas as pd

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from typing import Optional, List
from preprocessed import df_final


app = FastAPI()

@app.get("/search")
def search_trademarks(
    productName: Optional[str] = Query(
        None,
        alias="상품명",
        description="상품명 검색 (ex: 제이케)"
    ),    
    registerStatus: Optional[str] = Query(
        None,
        alias="등록 상태",
        description="등록 상태 (ex: 등록, 실효, 거절, 출원)"
    ),

    year: Optional[int] = Query(
        None,
        alias="출원 연도",
        description="출원 연도 (ex: 2000)"
    )
):
    # 원본 유지
    df_filtered = df_final.copy()
    
    # 부분 일치 검색
    if productName:
        df_filtered = df_filtered[df_filtered["productName"].str.contains(productName, na=False)]

    # 정확히 일치
    if registerStatus:
        df_filtered = df_filtered[df_filtered["registerStatus"] == registerStatus]
    
    # 정확히 일치하는 년도        
    if year:
        df_filtered = df_filtered[df_filtered["applicationDate"].str.startswith(str(year))]

    return df_filtered.to_dict(orient="records")