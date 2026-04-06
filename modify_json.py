import os
import json

# JSON 파일들이 있는 폴더
folder = "/home/csw/Desktop/유배지/anylabeling/labelme_jsons"

for filename in os.listdir(folder):
    if filename.endswith(".json"):
        json_path = os.path.join(folder, filename)

        # 파일명에서 .json 제거 → .jpg 붙이기
        base_name = os.path.splitext(filename)[0]
        new_image_path = base_name + ".jpg"

        # JSON 로드
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # imagePath 수정
        data["imagePath"] = new_image_path

        # 저장 (덮어쓰기)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

print("완료")