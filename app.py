from urllib import request as requestURL
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from flask import jsonify
from flask import Flask, redirect, render_template, request, session
from flask_session import Session
import find_similar_question

# Configure application
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST", "GET"])
def analyze():
    if request.method == "GET":
        return render_template("index.html")

    if request.method == "POST":
        if request.form.get("sentence1"):
            sentence1 = request.form.get("sentence1")
        if request.form.get("sentence2"):
            sentence2 = request.form.get("sentence2")

        return render_template("analyze.html", res=dict(sentence1=sentence1,sentence2=sentence2,score=82.5))


@app.route("/similarities", methods=["GET"])
def similarities():
    return render_template("similarities.html")

@app.route("/_bert_viz", methods=["GET","POST"])
def _bert_viz():
    from bertviz import head_view
    from transformers import BertTokenizer, BertModel
    sentence1 = request.form.get("sentence1") if request.form.get("sentence1") else ""
    sentence2 = request.form.get("sentence2") if request.form.get("sentence2") else ""
    model_version = 'bert-base-uncased'
    do_lower_case = True
    model = BertModel.from_pretrained(model_version, output_attentions=True)
    tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)
    inputs = tokenizer.encode_plus(sentence1, sentence2, return_tensors='pt', add_special_tokens=True)
    token_type_ids = inputs['token_type_ids']
    input_ids = inputs['input_ids']
    attention = model(input_ids, token_type_ids=token_type_ids)[-1]
    input_id_list = input_ids[0].tolist()  # Batch index 0
    tokens = tokenizer.convert_ids_to_tokens(input_id_list)
    return jsonify(result=head_view(attention, tokens))

@app.route("/bert", methods=["GET","POST"])
def bert():
    return render_template("bert.html")


# sim_qf = find_similar_question.SimilarQuestionFinder()
@app.route('/_get_similarities', methods=['GET', 'POST'])
def _get_similarities():
    question = request.form.get("question") if request.form.get("question") else ""
    # answers = sim_qf.get_similars_for_question(question).values.tolist()
    if not question: jsonify(result=render_template("similarities_answers.html", answers=[]))
    answers = [["question",60]]
    return jsonify(result=render_template("similarities_answers.html", answers=answers))

if __name__ == '__main__':
    app.run(debug=True)