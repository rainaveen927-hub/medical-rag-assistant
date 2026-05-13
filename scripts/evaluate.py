import pandas as pd
from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu

from app.rag import retrieve_docs, generate_answer

df = pd.read_csv("app/data/ragcare_qa.csv")

references = []
predictions = []

for _, row in df.head(20).iterrows():

    query = (
        row.get("question")
        or row.get("Question")
        or row.get("input")
    )

    ground_truth = (
        row.get("answer")
        or row.get("Answer")
        or row.get("output")
    )

    docs = retrieve_docs(query)

    prediction = generate_answer(
        query,
        docs
    )

    references.append([ground_truth])
    predictions.append(prediction)

bleu = corpus_bleu(predictions, references)

scorer = rouge_scorer.RougeScorer(
    ['rouge1', 'rougeL'],
    use_stemmer=True
)

rouge_scores = []

for ref, pred in zip(references, predictions):
    score = scorer.score(ref[0], pred)
    rouge_scores.append(score)

print("BLEU Score:", bleu.score)

avg_rouge1 = sum(
    s["rouge1"].fmeasure
    for s in rouge_scores
) / len(rouge_scores)

print("Average ROUGE-1:", avg_rouge1)