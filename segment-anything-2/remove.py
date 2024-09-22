import os

# 파일이 저장된 디렉토리 경로
folder_path = './tempframe'

# 삭제할 기준 파일 이름 (숫자 형식인 경우)
threshold = 0

# 폴더의 파일들을 가져옴
files = os.listdir(folder_path)

# 파일 이름을 순차적으로 확인하고 기준 넘는 파일 삭제
for file in files:
    # 파일 이름에서 숫자 부분을 추출
    file_name, file_ext = os.path.splitext(file)
    
    # 파일이 숫자로만 구성된 경우에만 처리
    if file_name.isdigit() and file_ext == '.jpg':
        file_number = int(file_name)
        
        # 파일 번호가 threshold 값보다 크면 삭제
        if file_number > threshold:
            file_path = os.path.join(folder_path, file)
            os.remove(file_path)
            print(f"{file_path} 삭제 완료.")
