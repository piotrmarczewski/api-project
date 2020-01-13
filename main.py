from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from Database import Database
from sklearn import tree

db = Database('src/sejm_gov_pl_db.db')

# conn = sqlite3.connect("src/sejm_gov_pl_db.db")
# df = pd.read_sql_query("SELECT * FROM portraits;", db.connection())
df = db.read_as_pd("SELECT full_name, last_party, speech_raw FROM speech_data sd JOIN portraits po ON sd.id_=po.id_")
# print(df["speech_raw"][0:100])

vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7)
X = vectorizer.fit_transform(df["speech_raw"][1:100]).toarray()
print(X)

# cls = tree.DecisionTreeClassifier()
# cls.fit(df["speech_raw"][1:100], df["last_party"][1:100])
# cls.predict(df["speech_raw"][101:201])

# db.query("ALTER TABLE portraits ADD new_part_name TEXT;")
# for row in db.query('SELECT last_party, COUNT(*) FROM speech_data sd JOIN portraits po ON sd.id_=po.id_ GROUP BY last_party ORDER BY COUNT(*) desc'):
#     print(row)
# for row in db.query('UPDATE portraits SET new_part_name=REPLACE((SELECT new_part_name FROM portraits po1 WHERE id_=po1.id_),"\r","")'):
#     print(row)

# for row in db.query('SELECT last_party, new_part_name FROM portraits LIMIT 10'):
#     print(row)

# c = conn.cursor()
# c2 = conn.cursor()
#
# # t = ('%Adam Abramowicz%',)
# # for row in c.execute('SELECT party_section FROM portraits'):
# #     print(row)
# #     a = (row[1],)
# #     for row2 in c2.execute('SELECT count(*) FROM speech_data WHERE id_=?', a):
# #         print(row2)
#
#
# t = ('%polskie%',)
# for row in c.execute('SELECT last_party, COUNT(*) FROM speech_data sd JOIN portraits po ON sd.id_=po.id_ GROUP BY last_party ORDER BY COUNT(*) desc'):
#     print(row)
