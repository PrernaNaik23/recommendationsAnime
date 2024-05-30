import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

anime_path = r'C:\Users\prern\PycharmProjects\recommendationsAnime\myanimelist.csv'
anime = pd.read_csv(anime_path)
anime = anime.drop_duplicates(subset=['title'])
anime['synopsis'] = anime['synopsis'].str.replace('[Written by MAL Rewrite]', '', regex=False)
anime['genre'] = anime['genre'].fillna('')
anime['aired'] = pd.to_datetime(anime['aired'].str.split('to').str[0], errors='coerce')

genre_list = ['Action', 'Adventure', 'Cars', 'Comedy', 'Drama', 'Dementia', 'Demons', 'Ecchi',
              'Fantasy', 'Game', 'Harem', 'Hentai', 'Historical', 'Horror', 'Magic',
              'Martial Arts', 'Mecha', 'Military', 'Music', 'Mystery', 'Parody', 'Police',
              'Psychological', 'Romance', 'Samurai', 'School', 'Sci-Fi', 'Slice of Life', 'Space',
              'Sports', 'Super Power', 'Supernatural', 'Thriller', 'Vampire']

demographic_list = ['Josei', 'Kids', 'Seinen', 'Shoujo', 'Shoujo Ai', 'Shounen', 'Shounen Ai', 'Yaoi', 'Yuri']

ps = PorterStemmer()


def stemming(text_data):
    data_set = [ps.stem(word) for word in text_data.split()]
    return " ".join(data_set)


def filtered_anime_genre(anime_data, input_genre):
    input_demographic = next((genre for genre in input_genre if genre in demographic_list), None)
    if not input_demographic:
        return pd.DataFrame()

    anime_data['genre'] = anime_data['genre'].apply(lambda x: x.strip('[]').replace("'", "").split(', '))

    filtered_anime = anime_data[anime_data['genre'].apply(
        lambda x: input_demographic in x and any(demo in x for demo in demographic_list)
    )]

    return filtered_anime


name = input('Enter anime name: ').strip().lower()

try:
    input_anime_genre = anime[anime['title'].str.lower() == name]['genre'].apply(
        lambda x: x.strip('[]').replace("'", "").split(', ')).values[0]

except IndexError:
    print("No similar anime was found!")
    input_anime_genre = []

if input_anime_genre:
    filtered_anime = filtered_anime_genre(anime, input_anime_genre).copy()
    filtered_anime['synopsis'] = filtered_anime['synopsis'].fillna('').apply(stemming)

    weight = TfidfVectorizer(max_features=5000, stop_words='english')
    vec = weight.fit_transform(filtered_anime['synopsis']).toarray()
    cos_sim = cosine_similarity(vec)


    def recommend_anime(value, start_idx=0, batch_size=10):
        value = value.lower()
        anime_list = filtered_anime[filtered_anime['title'].str.lower().str.contains(value)]
        recommendations = []
        for index in anime_list.index:
            idx = filtered_anime.index.get_loc(index)
            similar_anime = sorted(enumerate(cos_sim[idx]), key=lambda x: x[1], reverse=True)[1:11]
            sorted_similar_titles = [filtered_anime.iloc[i[0]]['title'] for i in similar_anime]
            recommendations.extend(sorted_similar_titles)

        start = start_idx * batch_size
        end = start + batch_size
        return recommendations[start:end] if recommendations else ['No movies found']


    def sort_recommendations_by_date(recommendations, anime_data):
        anime_with_dates = {title: anime_data.loc[anime_data['title'] == title, 'aired'].values[0]
                            for title in recommendations}
        sorted_recommendations = sorted([title for title in recommendations],
                                        key=lambda x: x[1], reverse=True)
        return sorted_recommendations


    recommendations = recommend_anime(name)
    recommendations = sort_recommendations_by_date(recommendations, anime)


    def print_recommendations(recommendations):
        if recommendations:
            for idx, title in enumerate(recommendations, start=1):
                print(f"{idx}. {title}")
        else:
            print("No movies found.")


    print_recommendations(recommendations)
