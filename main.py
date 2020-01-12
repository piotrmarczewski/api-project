import sqlite3

from Database import Database

conn = sqlite3.connect('src/sejm_gov_pl_db.db', detect_types=sqlite3.PARSE_COLNAMES)
print(conn)

db = Database('src/sejm_gov_pl_db.db')
for row in db.query('SELECT last_party, COUNT(*) FROM speech_data sd JOIN portraits po ON sd.id_=po.id_ GROUP BY last_party ORDER BY COUNT(*) desc'):
    print(row)

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