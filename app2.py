#!/usr/bin/env python3
import random

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import json
import math
import time
from flask import Flask, request, jsonify
from flask import render_template

# Redis connection
import redis
# the "Primary connection string" in the "Access keys"
# redis_passwd = "Oqt4OeR7RtTHxcdy8FjyiVasZcQIdhukpAzCaAymnnM="
# redis_host = "my-learn1.redis.cache.windows.net"
Q11_redis = "No"
Q12_redis = "No"
redis_passwd = "hphztr2iFgC9aggEOwbper0HbdXrT5x6uAzCaEPGjI8="
redis_host = "76001.redis.cache.windows.net"
cache = redis.StrictRedis(
            host=redis_host, port=6380,
            db=0, password=redis_passwd,
            ssl=True,
        )

if cache.ping():
    print("pong")


# to delete all data in the cache
def purge_cache():
    try:
        # 使用 flushall 方法清除所有数据
        result = cache.flushall()
        return result
    except Exception as e:
        print(f"清除缓存失败：{str(e)}")
        return False

from azure.cosmos import CosmosClient

DB_CONN_STR = "AccountEndpoint=https://tutorial-uta-cse6332.documents.azure.com:443/;AccountKey=fSDt8pk5P1EH0NlvfiolgZF332ILOkKhMdLY6iMS2yjVqdpWx4XtnVgBoJBCBaHA8PIHnAbFY4N9ACDbMdwaEw==;"
db_client = CosmosClient.from_connection_string(conn_str=DB_CONN_STR)
database = db_client.get_database_client("tutorial")
amazon_reviews_container = database.get_container_client('reviews')
us_cities_container = database.get_container_client('us_cities')


app = Flask(__name__)


def fetch_database_us_cities(container, city_name=None, include_header=False, exact_match=False):
    """
    根据city取数据
    :param city_name: city属性，如果为None，则取所有
    :param include_header:
    :param exact_match: 完全匹配 False: 模糊匹配
    :return:
    """
    QUERY = "SELECT * from us_cities"
    params = None
    if city_name is not None:
        QUERY = "SELECT * FROM us_cities p WHERE p.city is not @city_name"
        params = [dict(name="@city_name", value=city_name)]
        if not exact_match:
            QUERY = "SELECT * FROM us_cities p WHERE p.city not like @city_name"

    headers = ["city", "lat", "lng", "country", "state", "population"]
    result = []

    # quickly fetch the result if it already in the cache
    if cache.exists(QUERY):
        result = json.loads(cache.get(QUERY).decode())
        print("cache hit: [{}]".format(QUERY))

    else:
        row_id = 0
        for item in container.query_items(
                query=QUERY, parameters=params, enable_cross_partition_query=True,
        ):
            row_id += 1
            line = [str(row_id)]
            for col in headers:
                line.append(item[col])
            result.append(line)

        # cache the result for future queries
        cache.set(QUERY, json.dumps(result))
        print("cache miss: [{}]".format(QUERY))

    if include_header:
        line = [x for x in headers]
        line.insert(0, "")
        result.insert(0, line)

    return result


def fetch_database_amazon_reviews(container, city_name=None, include_header=False, exact_match=False):
    global Q12_redis
    """
    根据city取数据
    :param city_name: city属性，如果为None，则取所有
    :param include_header:
    :param exact_match: 完全匹配 False: 模糊匹配
    :return:
    """
    QUERY = "SELECT * from reviews OFFSET 0 LIMIT 10000"


    headers = ["score", "city", "title", "review"]
    result = []
    query = "Q12 SELECT * from reviews"

    # quickly fetch the result if it already in the cache
    if cache.exists(query):

        Q12_redis = "Yes"
        result = json.loads(cache.get(query).decode())
        print("cache hit: [{}]".format(query))

    else:
        Q12_redis = "No"
        row_id = 0
        for item in container.query_items(
                query=QUERY, enable_cross_partition_query=True
        ):
            row_id += 1
            line = [str(row_id)]
            for col in headers:
                line.append(item[col])
            result.append(line)

        # cache the result for future queries
        cache.set(query, json.dumps(result))
        print("cache miss: [{}]".format(query))

    if include_header:
        line = [x for x in headers]
        line.insert(0, "")
        result.insert(0, line)

    return result


def fetch_data_us_cities(container, city_name=None, include_header=False, exact_match=False):
    return fetch_database_us_cities(container, city_name=city_name, include_header=include_header,
                                    exact_match=exact_match)
    # with open("us-cities.csv") as csvfile:
    #     csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    #     row_id = -1
    #     wanted_data = []
    #     for row in csvreader:
    #         row_id += 1
    #         if row_id == 0 and not include_header:
    #             continue
    #         line = []
    #         col_id = -1
    #         is_wanted_row = False
    #         if city_name is None:
    #             is_wanted_row = True
    #         for raw_col in row:
    #             col_id += 1
    #             col = raw_col.replace('"', '')
    #             line.append(col)
    #             if col_id == 0 and city_name is not None:
    #                 if not exact_match and city_name.lower() in col.lower():
    #                     is_wanted_row = True
    #                 elif exact_match and city_name.lower() == col.lower():
    #                     is_wanted_row = True
    #         if is_wanted_row:
    #             if row_id > 0:
    #                 line.insert(0, "{}".format(row_id))
    #             else:
    #                 line.insert(0, "")
    #             wanted_data.append(line)
    # return wanted_data


def fetch_data_amazon_reviews(container, city_name=None, include_header=False, exact_match=False):
    return fetch_database_amazon_reviews(container, city_name=city_name, include_header=include_header,
                                         exact_match=exact_match)


# 加载停用词列表
with open('stopwords.txt', 'r') as stopwords_file:
    stopwords = stopwords_file.read().splitlines()



@app.route('/stat/knn_reviews', methods=['GET'])
def knn_reviews_stat():
    start_time = time.time()

    # 获取查询参数
    num_classes = int(request.args.get('classes', 6))
    k_param = int(request.args.get('k', 3))
    words_limit = int(request.args.get('words', 10))

    # 获取所有城市的坐标和评分数据
    print("获取城市信息中。。。")
    cities_items = fetch_data_us_cities(us_cities_container)
    print("获取评分信息中。。。")
    reviews_items = fetch_data_amazon_reviews(amazon_reviews_container)
    print("获取信息完成！")


    # 创建城市数组
    cities = [{'city': item[1], 'x': float(item[2]), 'y': float(item[3]), 'population': float(item[-1])} for item in
              cities_items]

    # 创建评分数组
    reviews = [{'city': item[2], 'score': float(item[1]), 'review': item[-1]} for item in reviews_items]

    # 获取训练集的 x 和 y 坐标
    train_data = np.array([[city['x'], city['y']] for city in cities[:num_classes]])
    # 获取预测集的 x 和 y 坐标
    predict_data = np.array([[city['x'], city['y']] for city in cities[num_classes:]])
    # 生成训练集标签
    train_labels = [i for i, _ in enumerate(range(num_classes))]

    # 使用 KNN 聚类算法, p=2为欧式距离
    knn = KNeighborsClassifier(n_neighbors=k_param, p=2).fit(train_data, train_labels)
    predict_labels = knn.predict(predict_data)
    print("聚类完成！")
    # 合并训练集和测试集的标签
    labels = np.concatenate((train_labels, predict_labels))
    # 初始化结果字典
    result = {"clusters": []}

    # 处理每个聚类
    for i in range(num_classes):
        print("处理第{}个聚类".format(i))
        # 获取该类别的城市
        cities_in_class = [cities[j] for j in range(len(cities)) if labels[j] == i]
        scores_in_class = [score['score'] for score in reviews if
                           score['city'] in [city['city'] for city in cities_in_class]]

        # 计算权重
        weights = [city['population'] / sum([city['population'] for city in cities_in_class])
                   for city in cities_in_class]

        # 计算加权平均分数
        weighted_avg_score = sum([score * weight for score, weight in zip(scores_in_class, weights)])

        # 找到类别中心点, 训练集中的城市即为中心城市
        # center_city = cities[i]
        center_city_coordinates = np.median(np.array([[city['x'], city['y']] for city in cities_in_class]), axis=0)
        center_city_coordinates = {'x': center_city_coordinates[0], 'y': center_city_coordinates[1]}

        # 找到最接近中心坐标的城市
        closest_city = min(cities_in_class, key=lambda city: np.sqrt(
            (city['x'] - center_city_coordinates['x']) ** 2 + (city['y'] - center_city_coordinates['y']) ** 2))
        center_city = closest_city['city']
        state = None
        # 搜州名
        Query = "SELECT * FROM us_cities p WHERE p.city = @city_name"
        params = [dict(name="@city_name", value=center_city)]
        for item in us_cities_container.query_items(
                query=Query, parameters=params, enable_cross_partition_query=True,
        ):
            state = item["state"]

        # 获取该类别的评论
        reviews_in_class = [review['review'] for review in reviews if
                            review['city'] in [city['city'] for city in cities_in_class]]

        # 处理评论，计算单词频率
        all_text = ' '.join(reviews_in_class).lower()
        words = all_text.split()
        # 去除停用词
        with open("stopwords.txt", "r") as stopword_file:
            stopwords = set(stopword_file.read().split())
        # 处理评论，去除停用词
        words = [word for word in words if word not in stopwords]
        word_counts = {word: words.count(word) for word in set(words)}

        # 按频率降序排序
        sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

        # 截取指定数量的单词
        top_words = [{"term": term, "popularity": count} for term, count in sorted_word_counts[:words_limit]]

        # 将结果添加到返回字典中
        result["clusters"].append({
            "center_city": center_city,
            "state": state,
            "weighted_avg_score": weighted_avg_score,
            "num_cities": len(cities_in_class),
            "top_words": top_words,
            "total_population": sum([city['population'] for city in cities_in_class])
        })

    # 计算处理时间
    processing_time = round((time.time() - start_time) * 1000, 2)
    result["processing_time"] = processing_time
    result["redisq12"] = Q12_redis

    return jsonify(result)


def q10_fetch_database(city_name = None, page = 0, page_size = 50):
    global Q11_redis
    cities_container = database.get_container_client("us_cities")
    QUERY = "SELECT * FROM us_cities p WHERE p.city = @city_name"
    params = [dict(name="@city_name", value=city_name)]
    row_id = 0
    city_headers = ["city", "lat", "lng", "country", "state", "population"]
    result = []
    for item in cities_container.query_items(
            query=QUERY, parameters=params, enable_cross_partition_query=True,
    ):
        print(item)
        row_id += 1
        line = [str(row_id)]
        for col in city_headers:
            line.append(item[col])
        result.append(line)
    city_lat = float(result[0][2])
    city_lng = float(result[0][3])
    print(result)

    QUERY = "SELECT * FROM us_cities p WHERE p.city != @city_name"
    params = [dict(name="@city_name", value=city_name)]

    query = "SELECT * FROM us_cities p WHERE p.city != " + city_name

    if cache.exists(query):
        Q11_redis = "Yes"
        result = json.loads(cache.get(query).decode())
        print("cache hit: [{}]".format(query))
        # 取出page_size个数据
        result = result[page * page_size: (page + 1) * page_size]
        return jsonify(result)

    else:
        Q11_redis = "No"
        result = []
        for item in cities_container.query_items(
                query=QUERY, parameters=params, enable_cross_partition_query=True,
        ):
            # 计算欧拉距离
            X1 = float(item["lat"])
            Y1 = float(item["lng"])
            distance = math.sqrt((city_lat - X1) ** 2 + (city_lng - Y1) ** 2)
            # distance保留四位小数
            distance = round(distance, 4)

            line = {
                "city": item["city"],
                "lat": item["lat"],
                "lng": item["lng"],
                "country": item["country"],
                "state": item["state"],
                "population": item["population"],
                "distance": distance
            }
            result.append(line)
        # 将result按照distance升序排序
        result = sorted(result, key=lambda x: x["distance"])


        # cache the result for future queries
        cache.set(query, json.dumps(result))
        print("cache miss: [{}]".format(query))
        # 取出page_size个数据
        result = result[page * page_size: (page + 1) * page_size]
        return jsonify(result)


def q10_fetch_database_new(city_name = None, state = None, page = 0, page_size = 50):
    cities_container = database.get_container_client("us_cities")
    QUERY = "SELECT * FROM us_cities p WHERE p.city = @city_name AND p.state = @state_name"
    # 传入城市名和州名
    params = [
        dict(name="@city_name", value=city_name),
        dict(name="@state_name", value=state)
    ]
    row_id = 0
    city_headers = ["city", "lat", "lng", "country", "state", "population"]
    result = []
    for item in cities_container.query_items(
            query=QUERY, parameters=params, enable_cross_partition_query=True,
    ):
        row_id += 1
        line = [str(row_id)]
        for col in city_headers:
            line.append(item[col])
        result.append(line)
    city_lat = float(result[0][2])
    city_lng = float(result[0][3])
    QUERY = "SELECT * FROM us_cities p WHERE p.city != @city_name AND p.state != @state_name"
    params = [
        dict(name="@city_name", value=city_name),
        dict(name="@state_name", value=state)
    ]
    query = "SELECT * FROM us_cities p WHERE p.city != " + city_name + " AND p.state != " + state

    if cache.exists(query):
        Q11_redis = "Yes"
        result = json.loads(cache.get(query).decode())
        print("cache hit: [{}]".format(query))
        # 取出page_size个数据
        # result = result[page * page_size: (page + 1) * page_size]
        return jsonify(result)
    #
    else:
        result = []
        for item in cities_container.query_items(
                query=QUERY, parameters=params, enable_cross_partition_query=True,
        ):
            # 计算欧拉距离
            X1 = float(item["lat"])
            Y1 = float(item["lng"])
            distance = math.sqrt((city_lat - X1) ** 2 + (city_lng - Y1) ** 2)
            # distance保留四位小数
            distance = round(distance, 4)

            line = {
                "city": item["city"],
                "lat": item["lat"],
                "lng": item["lng"],
                "country": item["country"],
                "state": item["state"],
                "population": item["population"],
                "distance": distance
            }
            result.append(line)
        # 将result按照distance升序排序
        result = sorted(result, key=lambda x: x["distance"])


        # cache the result for future queries
        cache.set(query, json.dumps(result))
        print("cache miss: [{}]".format(query))
        # 取出page_size个数据
        # result = result[page * page_size: (page + 1) * page_size]
        return jsonify(result)

def q11_fetch_database_new(city_name = None, state = None, page = 0, page_size = 50):
    speed = 0
    cities_container = database.get_container_client("us_cities")
    QUERY = "SELECT * FROM us_cities p WHERE p.city = @city_name AND p.state = @state_name"
    # 传入城市名和州名
    params = [
        dict(name="@city_name", value=city_name),
        dict(name="@state_name", value=state)
    ]
    row_id = 0
    city_headers = ["city", "lat", "lng", "country", "state", "population"]
    result = []
    for item in cities_container.query_items(
            query=QUERY, parameters=params, enable_cross_partition_query=True,
    ):
        row_id += 1
        line = [str(row_id)]
        for col in city_headers:
            line.append(item[col])
        result.append(line)
    city_lat = float(result[0][2])
    city_lng = float(result[0][3])
    QUERY = "SELECT * FROM us_cities p WHERE p.city != @city_name AND p.state != @state_name"
    params = [
        dict(name="@city_name", value=city_name),
        dict(name="@state_name", value=state)
    ]
    query = "Q11 SELECT * FROM us_cities p WHERE p.city != " + city_name + " AND p.state != " + state

    if cache.exists(query):
        Q11_redis = "Yes"
        result = json.loads(cache.get(query).decode())
        print("cache hit: [{}]".format(query))
        # 取出page_size个数据
        # result = result[page * page_size: (page + 1) * page_size]
        return jsonify(result)
    #
    else:
        result = []
        for item in cities_container.query_items(
                query=QUERY, parameters=params, enable_cross_partition_query=True,
        ):
            print(speed)
            speed += 1
            # 当speed = 51时退出循环
            if speed == 51:
                break
            # 计算欧拉距离
            X1 = float(item["lat"])
            Y1 = float(item["lng"])
            distance = math.sqrt((city_lat - X1) ** 2 + (city_lng - Y1) ** 2)
            # distance保留四位小数
            distance = round(distance, 4)
            # 从reviews中取出该城市的平均得分
            city = item["city"]
            QUERY = "SELECT * FROM reviews p WHERE p.city = @city_name"
            params = [dict(name="@city_name", value=city)]
            score = 0
            count = 0
            for review_item in amazon_reviews_container.query_items(
                    query=QUERY, parameters=params, enable_cross_partition_query=True,
            ):
                score += int(review_item["score"])
                count += 1
            if count != 0:
                score = score / count
            else:
                score = 0
            # 将score保留两位小数
            score = round(score, 2)

            line = {
                "city": item["city"],
                "lat": item["lat"],
                "lng": item["lng"],
                "country": item["country"],
                "state": item["state"],
                "population": item["population"],
                "distance": distance,
                "score": score
            }
            result.append(line)
        # 将result按照distance升序排序
        result = sorted(result, key=lambda x: x["distance"])


        # cache the result for future queries
        cache.set(query, json.dumps(result))
        print("cache miss: [{}]".format(query))
        # 取出page_size个数据
        # result = result[page * page_size: (page + 1) * page_size]
        return jsonify(result)

# app = Flask(__name__)


@app.route("/", methods=['GET'])
def index():
    return render_template(
        'index.html'
    )

@app.route('/stat/closest_cities', methods=['GET'])
def closest_cities():
    # 获取查询参数
    city = request.args.get('city', default="")
    page = request.args.get('page', default=0, type=int)
    page_size = request.args.get('page_size', default=50, type=int)


    # 记录当前时间
    start_time = time.time()
    # 将返回结果存入文件
    data = q10_fetch_database(city_name=city, page=page, page_size=page_size)
    end_time = time.time()

    return jsonify(
        {
            "data": data.json,
            "time": round((end_time - start_time)*1000,2),
            "redisq11": Q11_redis,
        }
    )

@app.route('/test', methods=['GET'])
def test():
    reviews_container = database.get_container_client("reviews")
    QUERY = "SELECT * FROM reviews LIMIT 20000"
    review_headers = ["score","city","title","review"]
    result = []
    for item in reviews_container.query_items(
            query=QUERY, enable_cross_partition_query=True,
    ):
        line = []
        for col in review_headers:
            line.append(item[col])
        # 将line以添加的方式写入文件
        with open('example.txt', 'a', encoding='utf-8') as f:
            f.write(str(line))

        result.append(line)
    return

# Flask 路由，用于清除缓存
@app.route('/flush_cache', methods=['GET'])
def flush_cache():
    success = purge_cache()
    if success:
        print("缓存清除成功")
        return jsonify({"result": "缓存清除成功"})
    else:
        print("缓存清除失败")
        return jsonify({"result": "缓存清除成功"})

@app.route('/stat/closest_cities_new', methods=['GET'])
def closest_cities_new():
    city = request.args.get('city', default="")
    state = request.args.get('state', default="")
    return q10_fetch_database_new(city_name=city, state=state)

@app.route('/stat/q11_closest_cities_new', methods=['GET'])
def q11_closest_cities_new():
    city = request.args.get('city', default="")
    state = request.args.get('state', default="")
    return q11_fetch_database_new(city_name=city, state=state)

@app.route("/new", methods=['GET'])
def index_new():
    return render_template(
        'index_new.html'
    )

@app.route("/details", methods=['POST'])
def details():
    data = request.form['data']
    # 对 data 进行处理，例如解析 JSON
    parsed_data = json.loads(data)

    # 渲染页面，传递数据到前端
    return render_template('details.html', data=parsed_data)

if __name__ == "__main__":

    app.run(host="127.0.0.1", port=5000, debug=True)
