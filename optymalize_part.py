import re

from Database import Database

db = Database('src/sejm_gov_pl_db.db')

df = db.read_as_pd("SELECT full_name, last_party, speech_raw FROM speech_data sd JOIN portraits po ON sd.id_=po.id_")

for key, party in df['last_party']:
    party = re.sub(r'\W', ' ', str(party))
    party = re.sub(r'\s+[a-zA-Z]\s+', ' ', party)
    party = re.sub(r'\^[a-zA-Z]\s+', ' ', party)
    party = re.sub(r'\s+', ' ', party, flags=re.I)
    party = party.lower()

    df['last_party'][key] = party

df.groupby(['Animal']).mean()