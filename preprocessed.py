import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import re
from korean_romanizer.romanizer import Romanizer
from g2pk import G2p
from itertools import chain
from collections import Counter
from datetime import timedelta, datetime

g2p = G2p()


# 'pNK': 'productName',  'pNE': 'productNameEng' 결측치 채우는 함수
def filled_pNK_pNE(row):
    pNK, pNE = row['pNK'], row['pNE']

    # 한글x 영문있으면 g2p로 한글 결측치 넣기
    if pd.isna(pNK) and pd.notna(pNE):
        try:
            pNK = g2p(pNE)
        except Exception:
            pNK = 'None'

    # 영어x 한글있으면 Romanizer로 영문화
    if pd.isna(pNE) and pd.notna(pNK):
        try:
            pNE = Romanizer(pNK).romanize()
        except Exception:
            pNE = 'None'

    # 둘다 None이면 None
    if pd.isna(pNK):
        pNK = 'None'
    if pd.isna(pNE):
        pNE = 'None'

    return pd.Series([pNK, pNE])


# 날짜 변환 함수(1970이전 처리)
def safe_to_datetime(val):
    try:
        return pd.to_datetime(val, format="%Y%m%d", errors='coerce')
    except Exception:
        return pd.NaT


# 날짜 포맷 함수
def format_date_for_json(x):
    if pd.isnull(x):
        return "날짜 없음"
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.strftime("%Y%m%d")
    for fmt in ("%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            dt = datetime.strptime(str(x), fmt)
            return dt.strftime("%Y%m%d")
        except Exception:
            continue
    return "날짜 없음"


# --------------------------------------
# 결측치 처리 순서 : 1(pNK, pNE), 2(pD, rD), 3(pN, rN), 4(pCNL, pCDL), 5(aPMCL, aPSCL), 6(vCL)
# --------------------------------------

# 1단계: pNK,pNE
def fill_step1(df):
    df[['pNK', 'pNE']] = df.apply(filled_pNK_pNE, axis=1)
    return df 

# 2단계: pD, rD
def fill_step2(df):
    df['aD'] = df['aD'].apply(safe_to_datetime)
    df['pD'] = df['pD'].apply(safe_to_datetime)
    df['rD_flat'] = df['rD'].apply(
        lambda x: safe_to_datetime(x[0]) if isinstance(x, list) and x else pd.NaT
    )

    mask = df['rS'].isin(['등록', '실효'])

    valid_pD = df[mask & df['aD'].notnull() & df['pD'].notnull() & (df['pD'] > df['aD'])]
    valid_rD = df[mask & df['aD'].notnull() & df['rD_flat'].notnull() & (df['rD_flat'] > df['aD'])]

    mean_pub_delay = (valid_pD['pD'] - valid_pD['aD']).mean()
    mean_reg_delay = (valid_rD['rD_flat'] - valid_rD['aD']).mean()

    if pd.isnull(mean_pub_delay) or mean_pub_delay == pd.Timedelta(0):
        raise ValueError("유효한 평균 공고 지연일 계산에 실패했습니다.")
    if pd.isnull(mean_reg_delay) or mean_reg_delay == pd.Timedelta(0):
        raise ValueError("유효한 평균 등록 지연일 계산에 실패했습니다.")

    if 'iE' not in df.columns:
        df['iE'] = False

    for idx, row in df[mask].iterrows():
        aD = row['aD']
        if pd.isna(row['pD']) and not pd.isnull(aD):
            df.at[idx, 'pD'] = aD + mean_pub_delay
            df.at[idx, 'iE'] = True

        if not isinstance(row['rD'], list) or len(row['rD']) == 0 or all(pd.isnull(x) for x in row['rD']):
            est_rD = aD + mean_reg_delay if not pd.isnull(aD) else pd.NaT
            df.at[idx, 'rD'] = [est_rD]
            df.at[idx, 'iE'] = True

    return df

# 3단계: pN, rN
def fill_step3(df):
    mask_other = df['rS'].isin(['거절', '출원']) & (
        df['pN'].isnull() |
        df['pD'].isnull() |
        df['rN'].apply(lambda x: not isinstance(x, list) or len(x) == 0) |
        df['rD'].apply(lambda x: not isinstance(x, list) or len(x) == 0)
    )

    for idx, row in df[mask_other].iterrows():
        if pd.isna(row['pN']):
            df.at[idx, 'pN'] = ['공고 없음']
            df.at[idx, 'iE'] = True
        if pd.isna(row['pD']):
            df.at[idx, 'pD'] = pd.NaT
            df.at[idx, 'iE'] = True
        if not isinstance(row['rN'], list) or len(row['rN']) == 0:
            df.at[idx, 'rN'] = ['등록 없음']
            df.at[idx, 'iE'] = True
        if not isinstance(row['rD'], list) or len(row['rD']) == 0:
            df.at[idx, 'rD'] = [pd.NaT]
            df.at[idx, 'iE'] = True

    df['rD_flat'] = df['rD'].apply(
        lambda x: safe_to_datetime(x[0]) if isinstance(x, list) and x else pd.NaT
    )
    return df

# 4단계: pCNL, pCDL
def fill_step4(df):
    for idx, row in df.iterrows():
        if not isinstance(row['pCNL'], list) or len(row['pCNL']) == 0:
            df.at[idx, 'pCNL'] = ['우선권 없음']
            df.at[idx, 'iE'] = True
        if not isinstance(row['pCDL'], list) or len(row['pCDL']) == 0:
            df.at[idx, 'pCDL'] = ['날짜 없음']
            df.at[idx, 'iE'] = True
    return df

# 5단계: aPMCL, aPSCL 
def fill_step5(df):
    df['brandCode'] = df['aN'].astype(str).str[:2]
    df['mainCode'] = df['aPMCL'].apply(lambda x: x[0] if isinstance(x, list) and x else '결측')
    df['subCodeCount'] = df['aPSCL'].apply(lambda x: len(x) if isinstance(x, list) else 0)

    pmcl_map = (
        df[df['aPMCL'].notnull()]
        .groupby('brandCode')['aPMCL']
        .apply(lambda x: Counter([tuple(i) for i in x if isinstance(i, list)]).most_common(1)[0][0])
        .to_dict()
    )
    pscl_map = (
        df[df['aPSCL'].notnull()]
        .groupby('brandCode')['aPSCL']
        .apply(lambda x: Counter([tuple(i) for i in x if isinstance(i, list)]).most_common(1)[0][0])
        .to_dict()
    )

    for idx, row in df.iterrows():
        bc = row['brandCode']
        if (not isinstance(row['aPMCL'], list)) or (len(row['aPMCL']) == 0):
            if bc in pmcl_map:
                df.at[idx, 'aPMCL'] = list(pmcl_map[bc])
                df.at[idx, 'iE'] = True
        if (not isinstance(row['aPSCL'], list)) or (len(row['aPSCL']) == 0):
            if bc in pscl_map:
                df.at[idx, 'aPSCL'] = list(pscl_map[bc])
                df.at[idx, 'iE'] = True

    fallback_pmcl = (
        df['aPMCL'].dropna().map(tuple).value_counts().idxmax()
    )

    df['aPMCL'] = df['aPMCL'].apply(
        lambda x: list(fallback_pmcl) if not isinstance(x, list) or len(x) == 0 else x
    )
    df.loc[df['aPMCL'].apply(lambda x: x == list(fallback_pmcl)), 'iE'] = True

    return df

# 6단계: vCL 결측치 처리
def fill_step6(df):
    df['vCL'] = df['vCL'].apply(
        lambda x: ['None'] if not isinstance(x, list) or len(x) == 0 else x
    )
    df.loc[df['vCL'].apply(lambda x: x == ['None']), 'iE'] = True
    return df

# 전체 실행
def run_steps(df):
    df['iE'] = False  # 초기화
    df = fill_step1(df)
    df = fill_step2(df)
    df = fill_step3(df)
    df = fill_step4(df)
    df = fill_step5(df)
    df = fill_step6(df)
    return df


# 컬럼 재매핑 및 type처리 최종return설정
def to_final_output(df):
    # 컬럼명 매핑
    col_final_map = {
        'pNK': 'productName',
        'pNE': 'productNameEng',
        'aN': 'applicationNumber',
        'aD': 'applicationDate',
        'rS': 'registerStatus',
        'pN': 'publicationNumber',
        'pD': 'publicationDate',
        'rN': 'registrationNumber',
        'rD': 'registrationDate',
        'pCNL': 'priorityClaimNumList',
        'pCDL': 'priorityClaimDateList',
        'aPMCL': 'asignProductMainCodeList',
        'aPSCL': 'asignProductSubCodeList',
        'vCL': 'viennaCodeList',
        'iE': 'isEstimated'
    }

    # 이름 변경
    reverse_map = {k: v for k, v in col_final_map.items()}
    df_final = df.rename(columns=reverse_map).copy()

    # 불필요 컬럼 제거
    drop_cols = ['rD_flat', 'brandCode', 'mainCode', 'subCodeCount']
    df_final.drop(columns=[col for col in drop_cols if col in df_final.columns], inplace=True)

    # 날짜 포맷 정리
    date_cols = ['applicationDate', 'publicationDate']
    for col in date_cols:
        df_final[col] = df_final[col].apply(format_date_for_json)

    df_final['registrationDate'] = df_final['registrationDate'].apply(
        lambda lst: [format_date_for_json(d) for d in lst] if isinstance(lst, list) else ["날짜 없음"]
    )

    # 숫자 ID 문자열 변환
    df_final['applicationNumber'] = df_final['applicationNumber'].astype(str)
    df_final['publicationNumber'] = df_final['publicationNumber'].astype(str)

    return df_final

# --------------------------------------------------------- #

df = pd.read_json('trademark_sample.json')
c_map = {
    'pNK': 'productName',
    'pNE': 'productNameEng',
    'aN': 'applicationNumber',
    'aD': 'applicationDate',
    'rS': 'registerStatus',
    'pN': 'publicationNumber',
    'pD': 'publicationDate',
    'rN': 'registrationNumber',
    'rD': 'registrationDate',
    'rPN': 'registrationPubNumber',
    'rPD': 'registrationPubDate',
    'iRD': 'internationalRegDate',
    'iRN': 'internationalRegNumbers',
    'pCNL': 'priorityClaimNumList',
    'pCDL': 'priorityClaimDateList',
    'aPMCL': 'asignProductMainCodeList',
    'aPSCL': 'asignProductSubCodeList',
    'vCL': 'viennaCodeList'
}
df_s = df.rename(columns={v: k for k, v in c_map.items()})

# 사용하지 않는 columns 제거
cols_to_drop = ['rPN', 'rPD', 'iRD', 'iRN']
df_s.drop(columns=cols_to_drop, inplace=True)


# 날짜기준 정렬
df_ss = df_s.sort_values(by='aD')

# 결측치 처리 완료 후
df_clean = run_steps(df_s)

# 최종 출력용 데이터프레임 변환
df_final = to_final_output(df_clean)

# 저장
#df_final.to_json("home_final_sample.json", orient="records", force_ascii=False, indent=2)

# main.py import
__all__ = ["df_final"]



