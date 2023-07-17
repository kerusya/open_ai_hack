from flask import Flask, request, jsonify, render_template
import openai

app = Flask(__name__)

# Set up OpenAI API credentials
openai.api_key = "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"


@app.route('/')
def home():
    return render_template('index.html')

# Define endpoint for generating text
@app.route('/generate_text', methods=['POST'])
def generate_text():
    # Get prompt from request body

#     prompt = request.json['prompt']
    payload = {'prompt': str(request.form['area'])}
    # Generate text using OpenAI GPT API
    def get_completion(prompt, model="gpt-3.5-turbo"):
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message["content"] 

    # Return generated text as JSON response
    return render_template('index.html',prediction_text= get_completion(payload['prompt'])) 
#     return jsonify({'generated_text': get_completion(payload['prompt'])})

if __name__ == '__main__':
#     app.debug = True
#     app.run()
    app.run(host='0.0.0.0')
