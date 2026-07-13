# 📰 Fake News Detection — RNN vs LSTM vs GRU

A deep learning project that classifies news articles as **Fake** or **Real**, comparing a TF-IDF + Logistic Regression baseline against SimpleRNN, LSTM, and GRU models. Includes a deployable Streamlit app for real-time predictions.

## Demo

Enter an article's title and body text, and the app returns a **Fake / Real** prediction with a confidence score.
<img width="987" height="820" alt="image" src="https://github.com/user-attachments/assets/a068ff48-7e94-42f2-8bc1-d92cd2abe43e" />
<img width="1243" height="857" alt="Screenshot 2026-07-13 141539" src="https://github.com/user-attachments/assets/7894b160-b103-40b5-a4c2-51a4067a5f17" />


## Dataset

[ISOT Fake News Dataset](https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/) — ~44,900 news articles:
- `Fake.csv` — 23,481 fake articles
- `True.csv` — 21,417 real articles (Reuters)

Each row contains `title`, `text`, `subject`, and `date`.

> **Note on data leakage:** the `subject` and `date` columns almost perfectly separate fake vs. real in this dataset (fake articles only use subjects like `News`/`politics`; real ones only use `politicsNews`/`worldnews`). Both columns are dropped during preprocessing so the model learns from article *content*, not metadata shortcuts.

## Project Structure

```
├── Fake_News_Detection_RNN_LSTM_GRU.ipynb   # Full training notebook
├── app.py                                    # Streamlit inference app
├── fake_news_model.h5                        # Trained GRU model
├── tokenizer.pkl                              # Fitted Keras tokenizer
├── requirements.txt                           # Dependencies
└── README.md
```

## Approach

1. **Preprocessing** — combine title + body text, lowercase, strip HTML/URLs/punctuation, drop duplicates.
2. **Baseline** — TF-IDF (unigrams + bigrams) + Logistic Regression.
3. **Deep learning** — tokenize and pad sequences (`max_len=300`, vocab size 20,000), then train three architectures with identical Embedding → recurrent layer → Dropout → Dense → Sigmoid structure for a fair comparison:
   - SimpleRNN
   - LSTM
   - GRU
4. **Evaluation** — accuracy, precision, recall, F1, and confusion matrices on a held-out test set.

## Results

| Model                        | Accuracy | Precision | Recall | F1     |
|-------------------------------|----------|-----------|--------|--------|
| Logistic Regression (TF-IDF)  | ~99%     | 96        | 96     | 96    |
| SimpleRNN                     | 97        | 96        | 93     | 94     |
| LSTM                          | 97        | 95        | 96      | 95      |
| **GRU (deployed model)**      | **~98.2%** | ~98.3%  | ~98.1% | ~98.2% |



The deployed GRU model achieves a balanced confusion matrix (roughly equal false positive / false negative rates), meaning it isn't systematically biased toward either class.

## Streamlit link:-
https://fakenewsdetection-jtusn5epfpmmjvnmnwumjv.streamlit.app/

## Retraining

Open `Fake_News_Detection_RNN_LSTM_GRU.ipynb` in Google Colab (GPU runtime recommended), upload `Fake.csv` and `True.csv`, and run all cells. This regenerates `fake_news_model.h5` and `tokenizer.pkl`.

## Limitations

- Trained on articles from **2016–2017**; it may not generalize well to current events, new sources, or satire.
- The model likely picks up on **writing style** differences between the dataset's fake sources and Reuters, rather than verifying factual claims — it is not a fact-checker.
- Performance on short inputs (a headline alone, no body) is less reliable.

## Tech Stack

- Python, TensorFlow / Keras
- scikit-learn (baseline model, metrics)
- Streamlit (deployment)

## License

MIT — feel free to use and adapt this project.
