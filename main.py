import sqlite3

conn = sqlite3.connect('sejm_gov_pl_db.db')
print(conn)

c = conn.cursor()

t = ('%Adam%',)
c.execute('SELECT * FROM portraits WHERE full_name LIKE ?', t)
print(c.fetchone())
