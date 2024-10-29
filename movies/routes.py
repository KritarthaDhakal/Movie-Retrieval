from movies import app
from flask import render_template, request
from movies.movieRetrieve import documents, queries, ranked_results_vsm, ranked_results_bim, ranked_results_unigram
from movies.movieRetrieve import ranking_function_vsm, ranking_function_bim, ranking_function_unigram
from movies.formatter import format_docname, extract_content
 

@app.route('/')
@app.route('/home/')
def home_page():
    return render_template('home.html')


@app.route('/docs/')
def docs_page():
    formatted_movies_name = []

    movies = list(documents.keys())
    for movie in movies:
        formatted_movies_name.append(format_docname(movie))
    return render_template('documents.html', documents=formatted_movies_name)


@app.route('/queries/')
def query_page():
    return render_template('queries.html', queries=enumerate(queries, start=1), show_watermark=True)


@app.route('/queries/<int:query_id>/')
def query_results(query_id):

    selected_query = queries[query_id - 1]

    vsm_retrieve = extract_content(documents, ranked_results_vsm.get(query_id, [])[:3])
    bim_retrieve = extract_content(documents, ranked_results_bim.get(query_id, [])[:3])
    unigram_retrieve = extract_content(documents, ranked_results_unigram.get(query_id, [])[:3])

    retrieved_list = [vsm_retrieve, bim_retrieve, unigram_retrieve]

    return render_template('query_results.html', queries=enumerate(queries, start=1), selected_query=selected_query, docs=retrieved_list)



@app.route('/search/')
def search_page():
    return render_template('search.html', show_watermark=True)



@app.route('/search/result/', methods=['POST'])
def search_results():
    query = request.form['query']

    vsm_retrieve = extract_content(documents, ranking_function_vsm(documents, [query])[1][:3])
    bim_retrieve = extract_content(documents, ranking_function_bim(documents, [query])[1][:3])
    unigram_retrieve = extract_content(documents, ranking_function_unigram(documents, [query])[1][:3])

    retrieved_list = [vsm_retrieve, bim_retrieve, unigram_retrieve]

    return render_template('search_results.html', search_query=query, docs=retrieved_list)

