import pandas as pd
import os
import traceback

def update_stock_list():
    """
    한국거래소(KIND)에서 제공하는 공식 상장회사 목록을 다운로드하여
    'apps/stock_list.csv' 파일로 저장합니다.
    """
    print("한국거래소(KIND)에서 공식 상장회사 목록을 다운로드합니다...")
    try:
        # KIND에서 제공하는 공식 상장법인 목록 URL
        # 이 URL은 Excel 파일을 다운로드합니다.
        url = 'https://kind.krx.co.kr/corpgeneral/corpList.do?method=download'
        
        # pandas의 read_html을 사용하여 웹페이지의 테이블을 직접 읽어옵니다.
        # KRX에서 반환하는 파일은 HTML 형식의 테이블일 수 있으며, read_html이 더 안정적입니다.
        # 인코딩은 'euc-kr' 또는 'cp949'로 설정해야 합니다.
        df = pd.read_html(url, header=0, encoding='euc-kr')[0]
        
        # '종목코드' 컬럼을 6자리 문자열로 포맷합니다.
        df['종목코드'] = df['종목코드'].astype(str).str.zfill(6)
        
        # 필요한 컬럼('회사명', '종목코드')만 선택하고, 컬럼명을 영문으로 변경합니다.
        df = df[['회사명', '종목코드']]
        df.rename(columns={'회사명': 'name', '종목코드': 'code'}, inplace=True)

        if df.empty:
            print("오류: KRX에서 종목 정보를 가져오지 못했습니다.")
            return

        # 'apps' 디렉토리가 없으면 생성
        if not os.path.exists('apps'):
            os.makedirs('apps')
            
        # CSV 파일로 저장
        output_path = './apps/stock_list.csv'
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"성공! 총 {len(df)}개의 종목을 '{output_path}' 파일에 저장했습니다.")

    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        traceback.print_exc()
        print("네트워크 연결 또는 URL 주소를 확인해주세요. 문제가 지속되면 방화벽이나 보안 설정을 확인해야 할 수 있습니다.")

if __name__ == "__main__":
    update_stock_list()
