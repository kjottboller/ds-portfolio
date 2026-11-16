import os
import re
import json
import datetime
import urllib.request
import urllib.error
from collections import OrderedDict
import pytest

# pytest src/movielens_analysis.py 

# служебные функции
# class Tests
# class Movies
# class Ratings
#     class Movies
#     class Users
# class Tags
# class Links


DATASET_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', 'datasets')
)


DEFAULT_MOVIES_PATH = os.path.join(DATASET_DIR, 'movies.csv')
DEFAULT_RATINGS_PATH = os.path.join(DATASET_DIR, 'ratings.csv')
DEFAULT_TAGS_PATH = os.path.join(DATASET_DIR, 'tags.csv')
DEFAULT_LINKS_PATH = os.path.join(DATASET_DIR, 'links.csv')


FAKE_IMDB_HTML = '''
<html>
<head>
<title>Toy Story (1995) - IMDb</title>
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "Movie",
  "name": "Toy Story",
  "description": "toys come alive",
  "duration": "PT81M",
  "director": [{"@type": "Person", "name": "John Lasseter"}],
  "aggregateRating": {"@type": "AggregateRating", "ratingValue": "8.3", "ratingCount": "1000000"}
}
</script>
</head>
<body>
<div>Budget $30,000,000</div>
<div>Cumulative Worldwide Gross $394,436,586</div>
</body>
</html>
'''


def _default_path(file_name):
    """возвращает путь к файлу в ../datasets"""
    return os.path.join(DATASET_DIR, file_name)


def _resolve_path(path_to_file, default_name):
    """пытается найти файл по переданному пути или в ../datasets"""
    if path_to_file is None:
        return _default_path(default_name)

    if os.path.isfile(path_to_file):
        return path_to_file

    candidate = _default_path(path_to_file)
    if os.path.isfile(candidate):
        return candidate

    return path_to_file


def _split_csv_line(line):
    """разбивает csv-строку на части с учетом кавычек"""
    parts = []
    current = ''
    in_quotes = False
    index = 0

    while index < len(line):
        symbol = line[index]
        if symbol == '"':
            if in_quotes and index + 1 < len(line) and line[index + 1] == '"':
                current += '"'
                index += 1
            else:
                in_quotes = not in_quotes
        elif symbol == ',' and not in_quotes:
            parts.append(current)
            current = ''
        else:
            current += symbol
        index += 1

    parts.append(current)
    return parts


def _read_csv_rows(path_to_file, expected_header, limit=1000):
    """читает первые limit строк данных после заголовка"""
    real_path = path_to_file
    if not os.path.isfile(real_path):
        raise FileNotFoundError(f'file not found: {path_to_file}')

    rows = []
    with open(real_path, 'r', encoding='utf-8') as file:
        header = _split_csv_line(file.readline().rstrip('\n'))
        if header != expected_header:
            raise ValueError(f'wrong file structure: {path_to_file}')

        for line_number, line in enumerate(file, start=1):
            if limit is not None and line_number > limit:
                break

            clean_line = line.rstrip('\n')
            if clean_line == '':
                continue

            values = _split_csv_line(clean_line)
            if len(values) != len(expected_header):
                raise ValueError(f'wrong row structure in file: {path_to_file}')

            row = {}
            for index, column in enumerate(expected_header):
                row[column] = values[index]
            rows.append(row)
    return rows


def _extract_year_from_title(title):
    """достает год выпуска из названия фильма"""
    match = re.search(r'\((\d{4})\)\s*$', title)
    if match is None:
        return None
    return int(match.group(1))


def _extract_year_from_timestamp(timestamp_value):
    """достает год из unix timestamp"""
    return datetime.datetime.fromtimestamp(
        int(timestamp_value),
        datetime.timezone.utc
    ).year


def _median(values):
    """считает медиану без statistics"""
    sorted_values = sorted(values)
    length = len(sorted_values)
    if length == 0:
        return 0

    middle = length // 2
    if length % 2 == 1:
        return sorted_values[middle]
    return (sorted_values[middle - 1] + sorted_values[middle]) / 2


def _variance(values):
    """считает дисперсию"""
    if len(values) == 0:
        return 0

    average_value = sum(values) / len(values)
    total = 0
    for value in values:
        total += (value - average_value) ** 2
    return total / len(values)


def _ordered_dict_from_pairs(pairs):
    """создает ordered dict из списка пар"""
    result = OrderedDict()
    for key, value in pairs:
        result[key] = value
    return result


def _sort_pairs_by_value_desc(pairs):
    """сортирует пары по значению по убыванию"""
    return sorted(pairs, key=lambda item: (-item[1], item[0]))


def _sort_pairs_by_key_asc(pairs):
    """сортирует пары по ключу по возрастанию"""
    return sorted(pairs, key=lambda item: item[0])


def _normalize_field_name(field_name):
    """приводит имя поля к одному виду"""
    normalized = str(field_name).strip().lower()
    mapping = {
        'director': 'director',
        'budget': 'budget',
        'cumulative worldwide gross': 'gross_worldwide',
        'gross': 'gross_worldwide',
        'runtime': 'runtime',
        'rating value': 'rating_value',
        'rating count': 'rating_count',
        'description': 'description',
        'page title': 'page_title',
        'page name': 'page_name',
        'movie title': 'movie_title',
        'imdb link': 'imdb_link',
        'imdb id': 'imdb_id',
        'status': 'status',
        'error type': 'error_type',
        'error message': 'error_message',
    }
    return mapping.get(normalized, normalized)


def _extract_ld_json(html):
    """ищет json-ld с информацией о фильме"""
    matches = re.findall(
        r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>',
        html,
        re.DOTALL
    )
    for raw_json in matches:
        cleaned = raw_json.strip()
        if cleaned == '':
            continue
        try:
            data = json.loads(cleaned)
        except Exception:
            continue

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get('@type') == 'Movie':
                    return item
        elif isinstance(data, dict) and data.get('@type') == 'Movie':
            return data
    return None


def _parse_money(text):
    """пытается превратить строку с деньгами в число"""
    if text is None:
        return None

    clean_text = str(text).replace(',', '').strip()
    match = re.search(r'(\d+(?:\.\d+)?)', clean_text)
    if match is None:
        return None

    try:
        return float(match.group(1))
    except ValueError:
        return None


def _parse_runtime(text):
    """пытается извлечь длительность фильма в минутах"""
    if text is None:
        return None

    iso_match = re.search(r'PT(?:(\d+)H)?(?:(\d+)M)?', str(text))
    if iso_match is not None:
        hours = 0
        minutes = 0
        if iso_match.group(1) is not None:
            hours = int(iso_match.group(1))
        if iso_match.group(2) is not None:
            minutes = int(iso_match.group(2))
        return hours * 60 + minutes

    minute_match = re.search(r'(\d+)\s*minute', str(text).lower())
    if minute_match is not None:
        return int(minute_match.group(1))

    return None


def _extract_text_value(html, field_label):
    """пытается достать поле по текстовой подписи"""
    patterns = [
        rf'{field_label}[^\d$€£¥]*([$€£¥]?\s?[\d,]+(?:\.\d+)?)',
        rf'{field_label}.*?([$€£¥]?\s?[\d,]+(?:\.\d+)?)',
    ]
    for pattern in patterns:
        match = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
        if match is not None:
            return match.group(1)
    return None


class Tests:
    """тесты  pytest"""

    @staticmethod
    def is_dictionary(data):
        return isinstance(data, dict)

    @staticmethod
    def is_ordered_dict(data):
        return isinstance(data, OrderedDict)

    @staticmethod
    def is_list_of_strings(data):
        if not isinstance(data, list):
            return False
        for item in data:
            if not isinstance(item, str):
                return False
        return True

    @staticmethod
    def is_list_of_lists(data):
        if not isinstance(data, list):
            return False
        for item in data:
            if not isinstance(item, list):
                return False
        return True

    @staticmethod
    def is_sorted_by_value_desc(data):
        items = list(data.items())
        index = 1
        while index < len(items):
            previous_item = items[index - 1]
            current_item = items[index]
            if previous_item[1] < current_item[1]:
                return False
            if previous_item[1] == current_item[1] and previous_item[0] > current_item[0]:
                return False
            index += 1
        return True

    @staticmethod
    def is_sorted_by_key_asc(data):
        items = list(data.items())
        index = 1
        while index < len(items):
            if items[index - 1][0] > items[index][0]:
                return False
            index += 1
        return True

    @staticmethod
    def _movies():
        return Movies(DEFAULT_MOVIES_PATH)

    @staticmethod
    def _ratings():
        return Ratings(DEFAULT_RATINGS_PATH, DEFAULT_MOVIES_PATH)

    @staticmethod
    def _ratings_movies():
        return Ratings.Movies(DEFAULT_RATINGS_PATH, DEFAULT_MOVIES_PATH)

    @staticmethod
    def _users():
        return Ratings.Users(DEFAULT_RATINGS_PATH, DEFAULT_MOVIES_PATH)

    @staticmethod
    def _tags():
        return Tags(DEFAULT_TAGS_PATH)

    @staticmethod
    def _links():
        return Links(DEFAULT_LINKS_PATH, DEFAULT_MOVIES_PATH)

    def test_movies_wrong_file(self):
        with pytest.raises(ValueError):
            Movies(DEFAULT_RATINGS_PATH)

    def test_movies_dist_by_release(self):
        result = self._movies().dist_by_release()
        assert self.is_ordered_dict(result)
        assert self.is_sorted_by_value_desc(result)

    def test_movies_dist_by_genres(self):
        result = self._movies().dist_by_genres()
        assert self.is_ordered_dict(result)
        assert self.is_sorted_by_value_desc(result)

    def test_movies_most_genres(self):
        result = self._movies().most_genres(10)
        assert self.is_ordered_dict(result)
        assert self.is_sorted_by_value_desc(result)

    def test_movies_movies_with_genre(self):
        result = self._movies().movies_with_genre('Drama', 10)
        assert self.is_list_of_strings(result)
        assert result == sorted(result)

    def test_ratings_wrong_file(self):
        with pytest.raises(ValueError):
            Ratings(DEFAULT_MOVIES_PATH, DEFAULT_MOVIES_PATH)

    def test_ratings_dist_by_year(self):
        result = self._ratings().dist_by_year()
        assert self.is_ordered_dict(result)
        assert self.is_sorted_by_key_asc(result)

    def test_ratings_dist_by_rating(self):
        result = self._ratings().dist_by_rating()
        assert self.is_ordered_dict(result)
        assert self.is_sorted_by_key_asc(result)

    def test_ratings_top_by_num_of_ratings(self):
        result = self._ratings().top_by_num_of_ratings(10)
        assert self.is_ordered_dict(result)
        assert self.is_sorted_by_value_desc(result)

    def test_ratings_top_by_ratings(self):
        result = self._ratings().top_by_ratings(10)
        assert self.is_ordered_dict(result)
        assert self.is_sorted_by_value_desc(result)

    def test_ratings_top_by_ratings_median(self):
        result = self._ratings().top_by_ratings(10, 'median')
        assert self.is_ordered_dict(result)
        assert self.is_sorted_by_value_desc(result)

    def test_ratings_top_controversial(self):
        result = self._ratings().top_controversial(10)
        assert self.is_ordered_dict(result)
        assert self.is_sorted_by_value_desc(result)

    def test_ratings_movies_nested(self):
        result = self._ratings_movies().top_by_num_of_ratings(5)
        assert self.is_ordered_dict(result)

    def test_users_wrong_file(self):
        with pytest.raises(ValueError):
            Ratings.Users(DEFAULT_MOVIES_PATH, DEFAULT_MOVIES_PATH)

    def test_users_dist_by_num_of_ratings(self):
        result = self._users().dist_by_num_of_ratings()
        assert self.is_ordered_dict(result)
        assert self.is_sorted_by_key_asc(result)

    def test_users_dist_by_ratings(self):
        result = self._users().dist_by_ratings('average')
        assert self.is_ordered_dict(result)
        assert self.is_sorted_by_key_asc(result)

    def test_users_dist_by_ratings_median(self):
        result = self._users().dist_by_ratings('median')
        assert self.is_ordered_dict(result)
        assert self.is_sorted_by_key_asc(result)

    def test_users_top_controversial(self):
        result = self._users().top_controversial(10)
        assert self.is_ordered_dict(result)
        assert self.is_sorted_by_value_desc(result)

    def test_users_most_active_users(self):
        result = self._users().most_active_users(10)
        assert self.is_ordered_dict(result)
        assert self.is_sorted_by_value_desc(result)

    def test_users_most_generous_users(self):
        result = self._users().most_generous_users(10)
        assert self.is_ordered_dict(result)
        assert self.is_sorted_by_value_desc(result)

    def test_tags_wrong_file(self):
        with pytest.raises(ValueError):
            Tags(DEFAULT_MOVIES_PATH)

    def test_tags_most_words(self):
        result = self._tags().most_words(10)
        assert self.is_ordered_dict(result)
        assert self.is_sorted_by_value_desc(result)

    def test_tags_longest(self):
        result = self._tags().longest(10)
        assert self.is_list_of_strings(result)

    def test_tags_most_words_and_longest(self):
        result = self._tags().most_words_and_longest(10)
        assert self.is_list_of_strings(result)

    def test_tags_most_popular(self):
        result = self._tags().most_popular(10)
        assert self.is_ordered_dict(result)
        assert self.is_sorted_by_value_desc(result)

    def test_tags_tags_with(self):
        result = self._tags().tags_with('space')
        assert self.is_list_of_strings(result)

    def test_links_wrong_file(self):
        with pytest.raises(ValueError):
            Links(DEFAULT_MOVIES_PATH, DEFAULT_MOVIES_PATH)

    def test_links_get_imdb_link(self):
        result = self._links().get_imdb_link('Toy Story (1995)')
        assert isinstance(result, str)
        assert result.startswith('https://www.imdb.com/title/tt')

    def test_links_missing_tmdb(self):
        result = self._links().missing_tmdb(10)
        assert self.is_list_of_strings(result)

    def test_links_get_imdb_failed_request(self, monkeypatch):
        def fake_urlopen(*args, **kwargs):
            raise urllib.error.URLError('test error')

        monkeypatch.setattr(urllib.request, 'urlopen', fake_urlopen)
        result = self._links().get_imdb('Toy Story (1995)')
        assert isinstance(result, dict)
        assert result['movie_title'] == 'Toy Story (1995)'
        assert result['status'] == 'request failed'

    def test_links_get_imdb_list(self, monkeypatch):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return FAKE_IMDB_HTML.encode('utf-8')

        def fake_urlopen(*args, **kwargs):
            return FakeResponse()

        monkeypatch.setattr(urllib.request, 'urlopen', fake_urlopen)
        links = self._links()
        result = links.get_imdb(
            ['Toy Story (1995)'],
            ['Director', 'Budget', 'Cumulative Worldwide Gross', 'Runtime']
        )
        assert self.is_list_of_lists(result)
        assert len(result[0]) == 5

    def test_links_top_directors(self, monkeypatch):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return FAKE_IMDB_HTML.encode('utf-8')

        def fake_urlopen(*args, **kwargs):
            return FakeResponse()

        monkeypatch.setattr(urllib.request, 'urlopen', fake_urlopen)
        links = self._links()
        links.get_imdb(['Toy Story (1995)'], ['Director', 'Budget', 'Runtime'])
        result = links.top_directors(10)
        assert self.is_ordered_dict(result)

    def test_links_most_expensive(self, monkeypatch):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return FAKE_IMDB_HTML.encode('utf-8')

        def fake_urlopen(*args, **kwargs):
            return FakeResponse()

        monkeypatch.setattr(urllib.request, 'urlopen', fake_urlopen)
        links = self._links()
        links.get_imdb(['Toy Story (1995)'], ['Budget'])
        result = links.most_expensive(10)
        assert self.is_ordered_dict(result)

    def test_links_most_profitable(self, monkeypatch):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return FAKE_IMDB_HTML.encode('utf-8')

        def fake_urlopen(*args, **kwargs):
            return FakeResponse()

        monkeypatch.setattr(urllib.request, 'urlopen', fake_urlopen)
        links = self._links()
        links.get_imdb(['Toy Story (1995)'], ['Budget', 'Cumulative Worldwide Gross'])
        result = links.most_profitable(10)
        assert self.is_ordered_dict(result)

    def test_links_longest(self, monkeypatch):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return FAKE_IMDB_HTML.encode('utf-8')

        def fake_urlopen(*args, **kwargs):
            return FakeResponse()

        monkeypatch.setattr(urllib.request, 'urlopen', fake_urlopen)
        links = self._links()
        links.get_imdb(['Toy Story (1995)'], ['Runtime'])
        result = links.longest(10)
        assert self.is_ordered_dict(result)

    def test_links_top_cost_per_minute(self, monkeypatch):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return FAKE_IMDB_HTML.encode('utf-8')

        def fake_urlopen(*args, **kwargs):
            return FakeResponse()

        monkeypatch.setattr(urllib.request, 'urlopen', fake_urlopen)
        links = self._links()
        links.get_imdb(['Toy Story (1995)'], ['Budget', 'Runtime'])
        result = links.top_cost_per_minute(10)
        assert self.is_ordered_dict(result)


class Movies:
    """analyzing data from movies.csv"""

    def __init__(self, path_to_the_file=None, limit=1000):
        # сохраняем данные из movies.csv в поле класса
        resolved_path = _resolve_path(path_to_the_file, 'movies.csv')
        rows = _read_csv_rows(resolved_path, ['movieId', 'title', 'genres'], limit)
        self.data = []
        for row in rows:
            self.data.append(
                {
                    'movieId': int(row['movieId']),
                    'title': row['title'],
                    'genres': row['genres'].split('|'),
                    'year': _extract_year_from_title(row['title']),
                }
            )

    def dist_by_release(self):
        # считаем фильмы по годам выпуска
        counts = {}
        for movie in self.data:
            if movie['year'] is None:
                continue
            if movie['year'] not in counts:
                counts[movie['year']] = 0
            counts[movie['year']] += 1
        return _ordered_dict_from_pairs(_sort_pairs_by_value_desc(list(counts.items())))

    def dist_by_genres(self):
        # считаем фильмы по жанрам
        counts = {}
        for movie in self.data:
            for genre in movie['genres']:
                if genre not in counts:
                    counts[genre] = 0
                counts[genre] += 1
        return _ordered_dict_from_pairs(_sort_pairs_by_value_desc(list(counts.items())))

    def most_genres(self, n):
        # ищем фильмы с наибольшим числом жанров
        result = []
        for movie in self.data:
            result.append((movie['title'], len(movie['genres'])))
        result = _sort_pairs_by_value_desc(result)
        return _ordered_dict_from_pairs(result[:n])

    def movies_with_genre(self, genre, n=10):
        # это дополнительный метод для быстрого просмотра жанра
        result = []
        for movie in self.data:
            if genre in movie['genres']:
                result.append(movie['title'])
        result.sort()
        return result[:n]


class _RatingsMoviesBase:
    """общая логика для анализа ratings по фильмам"""

    def __init__(self, path_to_the_ratings=None, path_to_the_movies=None, limit=1000):
        # соединяем ratings.csv и movies.csv в конструкторе
        resolved_ratings = _resolve_path(path_to_the_ratings, 'ratings.csv')
        resolved_movies = _resolve_path(path_to_the_movies, 'movies.csv')

        movie_rows = _read_csv_rows(resolved_movies, ['movieId', 'title', 'genres'], limit)
        rating_rows = _read_csv_rows(resolved_ratings, ['userId', 'movieId', 'rating', 'timestamp'], limit)

        movies_by_id = {}
        for row in movie_rows:
            movies_by_id[int(row['movieId'])] = row['title']

        self.data = []
        for row in rating_rows:
            movie_id = int(row['movieId'])
            if movie_id not in movies_by_id:
                continue
            self.data.append(
                {
                    'userId': int(row['userId']),
                    'movieId': movie_id,
                    'title': movies_by_id[movie_id],
                    'rating': float(row['rating']),
                    'timestamp': int(row['timestamp']),
                    'year': _extract_year_from_timestamp(row['timestamp']),
                }
            )

    def dist_by_year(self):
        # считаем оценки по годам выставления
        counts = {}
        for item in self.data:
            if item['year'] not in counts:
                counts[item['year']] = 0
            counts[item['year']] += 1
        return _ordered_dict_from_pairs(_sort_pairs_by_key_asc(list(counts.items())))

    def dist_by_rating(self):
        # считаем сколько раз встречается каждая оценка
        counts = {}
        for item in self.data:
            if item['rating'] not in counts:
                counts[item['rating']] = 0
            counts[item['rating']] += 1
        return _ordered_dict_from_pairs(_sort_pairs_by_key_asc(list(counts.items())))

    def top_by_num_of_ratings(self, n):
        # топ фильмов по числу оценок
        counts = {}
        for item in self.data:
            if item['title'] not in counts:
                counts[item['title']] = 0
            counts[item['title']] += 1
        return _ordered_dict_from_pairs(_sort_pairs_by_value_desc(list(counts.items()))[:n])

    def top_by_ratings(self, n, metric='average'):
        # топ фильмов по средней оценке или медиане
        grouped = {}
        for item in self.data:
            if item['title'] not in grouped:
                grouped[item['title']] = []
            grouped[item['title']].append(item['rating'])

        result = []
        for title, values in grouped.items():
            if metric == 'average':
                score = sum(values) / len(values)
            elif metric == 'median':
                score = _median(values)
            else:
                raise ValueError('metric should be average or median')
            result.append((title, round(score, 2)))

        return _ordered_dict_from_pairs(_sort_pairs_by_value_desc(result)[:n])

    def top_controversial(self, n):
        # топ фильмов по дисперсии оценок
        grouped = {}
        for item in self.data:
            if item['title'] not in grouped:
                grouped[item['title']] = []
            grouped[item['title']].append(item['rating'])

        result = []
        for title, values in grouped.items():
            result.append((title, round(_variance(values), 2)))
        return _ordered_dict_from_pairs(_sort_pairs_by_value_desc(result)[:n])


class Ratings(_RatingsMoviesBase):
    """analyzing data from ratings.csv"""

    def __init__(self, path_to_the_file=None, path_to_the_movies=None, limit=1000):
        super().__init__(path_to_the_file, path_to_the_movies, limit)

    class Movies(_RatingsMoviesBase):
        """анализ оценок по фильмам"""

    class Users(Movies):
        """анализ оценок по пользователям"""

        def dist_by_num_of_ratings(self):
            # распределение пользователей по числу оценок
            ratings_by_user = {}
            for item in self.data:
                if item['userId'] not in ratings_by_user:
                    ratings_by_user[item['userId']] = 0
                ratings_by_user[item['userId']] += 1

            distribution = {}
            for count in ratings_by_user.values():
                if count not in distribution:
                    distribution[count] = 0
                distribution[count] += 1
            return _ordered_dict_from_pairs(_sort_pairs_by_key_asc(list(distribution.items())))

        def dist_by_ratings(self, metric='average'):
            # распределение пользователей по их средней оценке или медиане
            ratings_by_user = {}
            for item in self.data:
                if item['userId'] not in ratings_by_user:
                    ratings_by_user[item['userId']] = []
                ratings_by_user[item['userId']].append(item['rating'])

            distribution = {}
            for values in ratings_by_user.values():
                if metric == 'average':
                    score = round(sum(values) / len(values), 2)
                elif metric == 'median':
                    score = round(_median(values), 2)
                else:
                    raise ValueError('metric should be average or median')

                if score not in distribution:
                    distribution[score] = 0
                distribution[score] += 1

            return _ordered_dict_from_pairs(_sort_pairs_by_key_asc(list(distribution.items())))

        def top_controversial(self, n):
            # топ пользователей по дисперсии их оценок
            ratings_by_user = {}
            for item in self.data:
                if item['userId'] not in ratings_by_user:
                    ratings_by_user[item['userId']] = []
                ratings_by_user[item['userId']].append(item['rating'])

            result = []
            for user_id, values in ratings_by_user.items():
                result.append((user_id, round(_variance(values), 2)))
            return _ordered_dict_from_pairs(_sort_pairs_by_value_desc(result)[:n])

        def most_active_users(self, n):
            # дополнительный метод: топ пользователей по активности
            ratings_by_user = {}
            for item in self.data:
                if item['userId'] not in ratings_by_user:
                    ratings_by_user[item['userId']] = 0
                ratings_by_user[item['userId']] += 1
            return _ordered_dict_from_pairs(_sort_pairs_by_value_desc(list(ratings_by_user.items()))[:n])

        def most_generous_users(self, n, metric='average'):
            # дополнительный метод: топ пользователей по мягкости оценок
            ratings_by_user = {}
            for item in self.data:
                if item['userId'] not in ratings_by_user:
                    ratings_by_user[item['userId']] = []
                ratings_by_user[item['userId']].append(item['rating'])

            result = []
            for user_id, values in ratings_by_user.items():
                if metric == 'average':
                    score = round(sum(values) / len(values), 2)
                elif metric == 'median':
                    score = round(_median(values), 2)
                else:
                    raise ValueError('metric should be average or median')
                result.append((user_id, score))
            return _ordered_dict_from_pairs(_sort_pairs_by_value_desc(result)[:n])


class Tags:
    """analyzing data from tags.csv"""

    def __init__(self, path_to_the_file=None, limit=1000):
        # сохраняем данные из tags.csv в поле класса
        resolved_path = _resolve_path(path_to_the_file, 'tags.csv')
        rows = _read_csv_rows(resolved_path, ['userId', 'movieId', 'tag', 'timestamp'], limit)
        self.data = []
        for row in rows:
            self.data.append(
                {
                    'userId': int(row['userId']),
                    'movieId': int(row['movieId']),
                    'tag': row['tag'],
                    'timestamp': int(row['timestamp']),
                }
            )

    def most_words(self, n):
        # топ тегов с наибольшим числом слов
        unique_tags = {}
        for item in self.data:
            unique_tags[item['tag']] = len(item['tag'].split())
        return _ordered_dict_from_pairs(_sort_pairs_by_value_desc(list(unique_tags.items()))[:n])

    def longest(self, n):
        # топ самых длинных уникальных тегов
        unique_tags = {}
        for item in self.data:
            unique_tags[item['tag']] = len(item['tag'])
        result = sorted(unique_tags.items(), key=lambda item: (-item[1], item[0]))
        tags = []
        for tag, _ in result[:n]:
            tags.append(tag)
        return tags

    def most_words_and_longest(self, n):
        # пересечение двух top-n списков
        most_words_tags = self.most_words(n)
        longest_tags = self.longest(n)
        result = []
        for tag in longest_tags:
            if tag in most_words_tags and tag not in result:
                result.append(tag)
        result.sort()
        return result

    def most_popular(self, n):
        # самые популярные теги
        counts = {}
        for item in self.data:
            if item['tag'] not in counts:
                counts[item['tag']] = 0
            counts[item['tag']] += 1
        return _ordered_dict_from_pairs(_sort_pairs_by_value_desc(list(counts.items()))[:n])

    def tags_with(self, word):
        # все уникальные теги, содержащие слово
        result = []
        lowered_word = word.lower()
        for item in self.data:
            tag = item['tag']
            if lowered_word in tag.lower() and tag not in result:
                result.append(tag)
        result.sort()
        return result


class Links:
    """analyzing data from links.csv"""

    def __init__(self, path_to_the_file=None, path_to_the_movies=None, limit=1000):
        # соединяем links.csv и movies.csv в конструкторе
        resolved_links = _resolve_path(path_to_the_file, 'links.csv')
        resolved_movies = _resolve_path(path_to_the_movies, 'movies.csv')

        link_rows = _read_csv_rows(resolved_links, ['movieId', 'imdbId', 'tmdbId'], limit)
        movie_rows = _read_csv_rows(resolved_movies, ['movieId', 'title', 'genres'], limit)

        movies_by_id = {}
        for row in movie_rows:
            movies_by_id[int(row['movieId'])] = row['title']

        self.data = []
        for row in link_rows:
            movie_id = int(row['movieId'])
            if movie_id not in movies_by_id:
                continue
            self.data.append(
                {
                    'movieId': movie_id,
                    'title': movies_by_id[movie_id],
                    'imdbId': row['imdbId'],
                    'tmdbId': row['tmdbId'],
                }
            )
        self.cache = {}

    def _find_by_title(self, title):
        # ищем фильм по названию
        for item in self.data:
            if item['title'] == title:
                return item
        raise ValueError(f'title not found: {title}')

    def _find_by_movie_id(self, movie_id):
        # ищем фильм по movieId
        for item in self.data:
            if item['movieId'] == movie_id:
                return item
        raise ValueError(f'movieId not found: {movie_id}')

    def _fetch_imdb_page(self, item):
        # делает один запрос к imdb и парсит несколько полей
        url = f"https://www.imdb.com/title/tt{item['imdbId']}/"
        result = {
            'movieId': item['movieId'],
            'movie_title': item['title'],
            'imdb_id': item['imdbId'],
            'imdb_link': url,
            'status': 'ok',
        }

        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(request, timeout=10) as response:
                html = response.read().decode('utf-8', errors='ignore')

            title_match = re.search(r'<title>(.*?)</title>', html, re.DOTALL)
            if title_match is not None:
                result['page_title'] = title_match.group(1).strip()

            ld_json = _extract_ld_json(html)
            if isinstance(ld_json, dict):
                result['page_name'] = ld_json.get('name')
                result['description'] = ld_json.get('description')
                if 'duration' in ld_json:
                    result['runtime'] = _parse_runtime(ld_json.get('duration'))

                aggregate_rating = ld_json.get('aggregateRating')
                if isinstance(aggregate_rating, dict):
                    if aggregate_rating.get('ratingValue') is not None:
                        result['rating_value'] = float(aggregate_rating.get('ratingValue'))
                    if aggregate_rating.get('ratingCount') is not None:
                        result['rating_count'] = int(str(aggregate_rating.get('ratingCount')).replace(',', ''))

                director = ld_json.get('director')
                if isinstance(director, list):
                    names = []
                    for person in director:
                        if isinstance(person, dict) and person.get('name') is not None:
                            names.append(person.get('name'))
                    if names:
                        result['director'] = ', '.join(names)
                elif isinstance(director, dict) and director.get('name') is not None:
                    result['director'] = director.get('name')

            budget_text = _extract_text_value(html, 'Budget')
            gross_text = _extract_text_value(html, 'Cumulative Worldwide Gross')
            runtime_text = _extract_text_value(html, 'Runtime')

            budget_value = _parse_money(budget_text)
            gross_value = _parse_money(gross_text)
            runtime_value = _parse_runtime(runtime_text)

            if budget_value is not None:
                result['budget'] = budget_value
            if gross_value is not None:
                result['gross_worldwide'] = gross_value
            if runtime_value is not None and 'runtime' not in result:
                result['runtime'] = runtime_value

        except Exception as error:
            result['status'] = 'request failed'
            result['error_type'] = error.__class__.__name__
            result['error_message'] = str(error)

        self.cache[item['movieId']] = result
        return result

    def _ensure_cached_by_title(self, title):
        # кэширует фильм по названию
        item = self._find_by_title(title)
        if item['movieId'] not in self.cache:
            return self._fetch_imdb_page(item)
        return self.cache[item['movieId']]

    def _ensure_cached_by_movie_id(self, movie_id):
        # кэширует фильм по movieId
        item = self._find_by_movie_id(movie_id)
        if item['movieId'] not in self.cache:
            return self._fetch_imdb_page(item)
        return self.cache[item['movieId']]

    def _warm_cache(self, count=10):
        # небольшой прогрев кэша для агрегатных методов
        index = 0
        while index < len(self.data) and index < count:
            item = self.data[index]
            if item['movieId'] not in self.cache:
                self._fetch_imdb_page(item)
            index += 1

    def get_imdb_link(self, title):
        # дополнительный метод: ссылка imdb по названию фильма
        item = self._find_by_title(title)
        return f"https://www.imdb.com/title/tt{item['imdbId']}/"

    def missing_tmdb(self, n):
        # дополнительный метод: фильмы без tmdb id
        result = []
        for item in self.data:
            if item['tmdbId'] == '':
                result.append(item['title'])
        result.sort()
        return result[:n]

    def get_imdb(self, list_of_movies, list_of_fields=None):
        # поддерживаем и sample-вариант, и удобный вариант с одним названием
        if list_of_fields is None and isinstance(list_of_movies, str):
            return self._ensure_cached_by_title(list_of_movies)

        result = []
        for movie in list_of_movies:
            if isinstance(movie, str):
                info = self._ensure_cached_by_title(movie)
            else:
                info = self._ensure_cached_by_movie_id(int(movie))

            row = [info['movieId']]
            for field_name in list_of_fields:
                normalized = _normalize_field_name(field_name)
                row.append(info.get(normalized))
            result.append(row)

        result.sort(key=lambda item: item[0], reverse=True)
        return result

    def top_directors(self, n):
        # топ режиссеров по числу фильмов в кэше
        if len(self.cache) == 0:
            self._warm_cache()

        counts = {}
        for info in self.cache.values():
            if info.get('status') != 'ok':
                continue
            directors = info.get('director')
            if directors is None:
                continue
            for director in str(directors).split(', '):
                if director not in counts:
                    counts[director] = 0
                counts[director] += 1
        return _ordered_dict_from_pairs(_sort_pairs_by_value_desc(list(counts.items()))[:n])

    def most_expensive(self, n):
        # топ фильмов по бюджету
        if len(self.cache) == 0:
            self._warm_cache()

        result = []
        for info in self.cache.values():
            if info.get('status') != 'ok':
                continue
            if info.get('budget') is not None:
                result.append((info['movie_title'], round(info['budget'], 2)))
        return _ordered_dict_from_pairs(_sort_pairs_by_value_desc(result)[:n])

    def most_profitable(self, n):
        # топ фильмов по прибыли
        if len(self.cache) == 0:
            self._warm_cache()

        result = []
        for info in self.cache.values():
            if info.get('status') != 'ok':
                continue
            if info.get('budget') is not None and info.get('gross_worldwide') is not None:
                profit = round(info['gross_worldwide'] - info['budget'], 2)
                result.append((info['movie_title'], profit))
        return _ordered_dict_from_pairs(_sort_pairs_by_value_desc(result)[:n])

    def longest(self, n):
        # топ фильмов по длительности
        if len(self.cache) == 0:
            self._warm_cache()

        result = []
        for info in self.cache.values():
            if info.get('status') != 'ok':
                continue
            if info.get('runtime') is not None:
                result.append((info['movie_title'], info['runtime']))
        return _ordered_dict_from_pairs(_sort_pairs_by_value_desc(result)[:n])

    def top_cost_per_minute(self, n):
        # топ фильмов по цене минуты экранного времени
        if len(self.cache) == 0:
            self._warm_cache()

        result = []
        for info in self.cache.values():
            if info.get('status') != 'ok':
                continue
            if info.get('budget') is not None and info.get('runtime') not in (None, 0):
                value = round(info['budget'] / info['runtime'], 2)
                result.append((info['movie_title'], value))
        return _ordered_dict_from_pairs(_sort_pairs_by_value_desc(result)[:n])


if __name__ == '__main__':
    pass
