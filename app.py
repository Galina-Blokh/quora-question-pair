from flask import Flask, render_template, request
import find_similar_question
print("[INFO] starting...")
app = Flask(__name__)
sim_qf = find_similar_question.SimilarQuestionFinder()
print("[OK] started")
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        question = request.form.get('question')
        print(str(sim_qf.get_similars_for_question(question).values.tolist()))
        return render_template('result.html', answers=sim_qf.get_similars_for_question(question).values.tolist())
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run()
