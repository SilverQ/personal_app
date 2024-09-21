from flask import Flask, request, jsonify
import ollama

app = Flask(__name__)

# Ollama 모델 이름 설정
model_name = "heegyu/EEVE-Korean-Insruct-10.8B-v1.0-GGUF"

# 모델 로드 (예시)
try:
    model = ollama.load_model(model_name)
except Exception as e:
    app.logger.error(f"Model load failed: {e}")

@app.route('/generate', methods=['POST'])
def generate_text():
    try:
        data = request.json
        prompt = data.get('prompt', '')
        # 여기에 응답 생성 로직 추가
        response = model.generate(prompt=prompt)
        return jsonify({'response': response['response']})
    except Exception as e:
        app.logger.error(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
