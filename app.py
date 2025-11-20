from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.get_json()
    user_message = data.get('message', '').strip()

    bot_response = f"Вы спросили: '{user_message}'\n\nЯ пока не умею искать мероприятия, но скоро научусь! 🚀"

    return jsonify({
        'success': True,
        'response': bot_response
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)