from flask import Flask, jsonify, request
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

try:
    model = pickle.load(open('artifacts/model.pkl', 'rb'))
    book_names = pickle.load(open('artifacts/book_names.pkl', 'rb'))
    final_rating = pickle.load(open('artifacts/final_rating.pkl', 'rb'))
    book_pivot = pickle.load(open('artifacts/book_pivot.pkl', 'rb'))
except FileNotFoundError as e:
    print(f"Error loading pickled files: {e}")
    exit(1)

def fetch_poster(suggestion):
    book_name = []
    ids_index = []
    poster_url = []

    for book_id in suggestion:
        book_name.append(book_pivot.index[book_id])

    for name in book_name[0]:
        try:
            ids = np.where(final_rating['title'] == name)[0][0]
            ids_index.append(ids)
        except IndexError as e:
            print(f"Error finding index: {e}")

    for idx in ids_index:
        url = final_rating.iloc[idx]['image_url']
        poster_url.append(url)

    return poster_url

def recommend_book(book_name):
    books_list = []
    try:
        book_id = np.where(book_pivot.index == book_name)[0][0]
        distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)

        poster_url = fetch_poster(suggestion)

        for i in range(len(suggestion)):
            books = book_pivot.index[suggestion[i]]
            for j in books:
                books_list.append(j)
    except IndexError as e:
        print(f"Error in recommend_book: {e}")
        return [], []

    return books_list, poster_url

@app.route('/')
def index():
    return 'Flask app running'

@app.route('/books')
def books():
    books = []
    for book in book_names:
        books.append(book)
    return jsonify({"books": books, "total" : len(books)})

@app.route('/detail', methods=["POST"])
def detail():
    search = request.get_json()
    books = []
    for book in final_rating:
        books.append(book)
    return jsonify({"books": books, "total" : len(books)})

@app.route('/recommend', methods=["POST"])
def recommend():
    data = request.get_json()
    recommended_books, poster_url = recommend_book(data["search"])
    return jsonify({"books": recommended_books, "posters": poster_url})


# if __name__ == "__main__":
#     app.run(debug=True)