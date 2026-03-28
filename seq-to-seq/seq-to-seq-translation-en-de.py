"""
seq-to-seq-translation-en-de.py
================================
End-to-end example of English → German machine translation using the
AWS SageMaker built-in Seq2Seq algorithm.
Pipeline
--------
  STEP 0  Prepare parallel corpus (English / German sentence pairs)
  STEP 1  Tokenise sentences into word lists
  STEP 2  Build source and target vocabularies  →  vocab.src.json / vocab.trg.json
  STEP 3  Encode sentences as padded integer token-ID sequences
  STEP 4  Write binary RecordIO protobuf files  (.rec)
  STEP 5  Upload data and vocab files to S3
  STEP 6  Configure and launch the SageMaker training job
  STEP 7  Deploy a real-time endpoint and run sample inference
Vocabulary file format
----------------------
The Seq2Seq container reads two JSON vocab files from the "vocab" S3 channel:
    vocab.src.json  — source language (English)
    vocab.trg.json  — target language (German)
Each file maps word strings to integer token IDs:

    {
      "<pad>": 0,
      "<eos>": 1,
      "<unk>": 2,
      "the":   3,
      "a":     4,
      ...
    }

The container builds its internal id_to_word lookup as:
    id_to_word[ int(float(value)) ] = key
Values must therefore equal the integer token IDs themselves.
RecordIO protobuf record layout
--------------------------------
Each record encodes one sentence pair:

    English  "the cat sat on the mat"
             → tokenise → [3, 12, 87, 9, 3, 63]
             → pad to max_seq_len_source (e.g. 50)
             stored in  record.features["source_ids"].int32_tensor

    German   "die Katze saß auf der Matte"
             → tokenise → [7, 18, 243, 14, 22, 115]
             → append <eos> (ID 1) → [7, 18, 243, 14, 22, 115, 1]
             → pad to max_seq_len_target (e.g. 50)
             stored in  record.features["target_ids"].int32_tensor
"""

import collections
import datetime
import json
import re
import struct

import boto3
from sagemaker.deserializers import JSONDeserializer
from sagemaker.estimator import Estimator
from sagemaker.image_uris import retrieve as get_image_uri
from sagemaker.inputs import TrainingInput
from sagemaker.serializers import JSONSerializer
from sagemaker.session import Session

try:
    from sagemaker.amazon.record_pb2 import Record
except ImportError:
    raise ImportError(
        "\n\nCould not import sagemaker.amazon.record_pb2.\n"
        "Install the required packages with:\n"
        "    pip install sagemaker protobuf\n"
    )


# ===========================================================================
# GLOBAL CONFIGURATION
# ===========================================================================

S3_BUCKET = "seq2seq-translation--148469447057-us-east-2-an"
S3_PREFIX = "seq2seq-en-de"
REGION    = "us-east-2"
ROLE_ARN  = "arn:aws:iam::148469447057:role/service-role/AmazonSageMaker-ExecutionRole-20260328T104911"
# GPU instance is required for training.
TRAINING_INSTANCE_TYPE  = "ml.g5.2xlarge"
INFERENCE_INSTANCE_TYPE = "ml.g5.2xlarge"
# Number of training instances.
# NOTE: The built-in Seq2Seq container only supports instance_count=1.
#       Increase this value only when using a custom distributed container.
INSTANCE_COUNT          = 1
# SNS topic that receives training-job lifecycle notifications.
# Set to None to disable SNS notifications.
SNS_TOPIC_ARN           = "arn:aws:sns:us-east-2:148469447057:seq2seq-training-notifications.fifo"
# ===========================================================================
# SEQUENCE AND VOCABULARY PARAMETERS
# ===========================================================================
# Sentences longer than these limits are truncated; shorter ones are padded.
MAX_SEQ_LEN_SOURCE = 50   # maximum English tokens per sentence
MAX_SEQ_LEN_TARGET = 50   # maximum German  tokens per sentence
# Only the top MAX_VOCAB_SIZE most-frequent words are kept in each vocabulary.
# Words outside this limit are replaced with the <unk> token at encode time.
MAX_VOCAB_SIZE = 30_000   # 30 k is a common choice for MT tasks
# ---------------------------------------------------------------------------
# Special token IDs — must be consistent across both vocab files and all .rec
# ---------------------------------------------------------------------------
PAD_ID      = 0   # <pad>  padding for sequences shorter than the maximum
EOS_ID      = 1   # <eos>  end-of-sequence, appended to every target sentence
UNK_ID      = 2   # <unk>  replaces words that are absent from the vocabulary
WORD_OFFSET = 3   # real word IDs begin at 3 (0–2 are reserved for specials)


# ===========================================================================
# LOCAL FILE PATHS
# ===========================================================================
TRAIN_FILE     = "translation-train.rec"
VAL_FILE       = "translation-val.rec"
SRC_VOCAB_FILE = "vocab.src.json"   # exact name expected by the container
TRG_VOCAB_FILE = "vocab.trg.json"   # exact name expected by the container

# RecordIO magic number — fixed by the MXNet / SageMaker specification.
RECORDIO_MAGIC = 0xCED7230A

# ===========================================================================
# STEP 0 — PREPARE PARALLEL CORPUS
# ===========================================================================
# Each entry is a (english_sentence, german_sentence) pair.
#
# For a production system, replace this inline list with a dataset loader:
#
#   with open("europarl-en.txt") as en_f, open("europarl-de.txt") as de_f:
#       raw_corpus = [(en.strip(), de.strip()) for en, de in zip(en_f, de_f)]
#
# Common freely available datasets:
#   WMT English-German  — 4.5 million sentence pairs
#   Europarl            — 2.0 million sentence pairs
#   Multi30K            — 30 thousand sentence pairs (good for quick experiments)
# ===========================================================================

RAW_CORPUS = [
    # (English source,                                          German target)

    # ── Original 20 ──────────────────────────────────────────────────────────
    ("the cat sat on the mat",                                "die Katze saß auf der Matte"),
    ("a dog is running in the park",                          "ein Hund läuft im Park"),
    ("she reads a book every evening",                        "sie liest jeden Abend ein Buch"),
    ("the sun rises in the east",                             "die Sonne geht im Osten auf"),
    ("we are learning machine translation",                   "wir lernen maschinelle Übersetzung"),
    ("the train arrives at eight o clock",                    "der Zug kommt um acht Uhr an"),
    ("please open the window",                                "bitte öffne das Fenster"),
    ("he works at a large company",                           "er arbeitet bei einem großen Unternehmen"),
    ("the children play in the garden",                       "die Kinder spielen im Garten"),
    ("I would like a cup of coffee",                          "ich möchte eine Tasse Kaffee"),
    ("the weather is very cold today",                        "das Wetter ist heute sehr kalt"),
    ("she speaks three languages fluently",                   "sie spricht drei Sprachen fließend"),
    ("the movie starts at nine o clock",                      "der Film beginnt um neun Uhr"),
    ("we visited the old castle yesterday",                   "wir haben gestern das alte Schloss besucht"),
    ("the library closes at six in the evening",              "die Bibliothek schließt um sechs Uhr abends"),
    ("he is writing a letter to his friend",                  "er schreibt einen Brief an seinen Freund"),
    ("the restaurant serves excellent food",                  "das Restaurant serviert ausgezeichnetes Essen"),
    ("they are travelling to Berlin next week",               "sie reisen nächste Woche nach Berlin"),
    ("the exam was more difficult than expected",             "die Prüfung war schwieriger als erwartet"),
    ("good morning how are you today",                        "guten Morgen wie geht es Ihnen heute"),

    # ── 21-50  Daily life ────────────────────────────────────────────────────
    ("I wake up early every morning",                         "ich wache jeden Morgen früh auf"),
    ("she drinks tea with milk",                              "sie trinkt Tee mit Milch"),
    ("the bus stops at the corner",                           "der Bus hält an der Ecke"),
    ("he is cooking dinner in the kitchen",                   "er kocht das Abendessen in der Küche"),
    ("the children are sleeping now",                         "die Kinder schlafen jetzt"),
    ("we need to buy groceries today",                        "wir müssen heute Lebensmittel kaufen"),
    ("the phone is ringing loudly",                           "das Telefon klingelt laut"),
    ("she is watching television in the living room",         "sie schaut im Wohnzimmer fern"),
    ("I take a shower every morning",                         "ich dusche mich jeden Morgen"),
    ("the post office is on the main street",                 "das Postamt befindet sich in der Hauptstraße"),
    ("he is reading the newspaper",                           "er liest die Zeitung"),
    ("the supermarket is open until midnight",                "der Supermarkt ist bis Mitternacht geöffnet"),
    ("she forgot her umbrella at home",                       "sie hat ihren Regenschirm zu Hause vergessen"),
    ("we are waiting for the taxi",                           "wir warten auf das Taxi"),
    ("the keys are on the table",                             "die Schlüssel liegen auf dem Tisch"),
    ("he locked the door before leaving",                     "er hat die Tür abgeschlossen bevor er ging"),
    ("the baby is crying in the bedroom",                     "das Baby weint im Schlafzimmer"),
    ("she is ironing her clothes",                            "sie bügelt ihre Kleidung"),
    ("I brush my teeth twice a day",                          "ich putze mir zweimal täglich die Zähne"),
    ("the garbage truck comes on Friday",                     "der Müllwagen kommt am Freitag"),
    ("he is fixing the broken chair",                         "er repariert den kaputten Stuhl"),
    ("she planted flowers in the garden",                     "sie hat Blumen im Garten gepflanzt"),
    ("the dog barked at the stranger",                        "der Hund bellte den Fremden an"),
    ("we cleaned the house yesterday",                        "wir haben gestern das Haus geputzt"),
    ("I need a new pair of shoes",                            "ich brauche ein neues Paar Schuhe"),
    ("the neighbours are very friendly",                      "die Nachbarn sind sehr freundlich"),
    ("she cut her finger while cooking",                      "sie hat sich beim Kochen den Finger geschnitten"),
    ("the milk is in the refrigerator",                       "die Milch ist im Kühlschrank"),
    ("he turned off the lights",                              "er hat das Licht ausgeschaltet"),
    ("we moved to a new apartment",                           "wir sind in eine neue Wohnung umgezogen"),

    # ── 51-100  Food and drink ───────────────────────────────────────────────
    ("I am very hungry right now",                            "ich bin jetzt sehr hungrig"),
    ("the bread is fresh from the bakery",                    "das Brot ist frisch von der Bäckerei"),
    ("she ordered a salad and soup",                          "sie bestellte einen Salat und eine Suppe"),
    ("the coffee is too hot to drink",                        "der Kaffee ist zu heiß zum Trinken"),
    ("we ate pizza for dinner last night",                    "wir haben letzte Nacht Pizza zum Abendessen gegessen"),
    ("he prefers red wine over white wine",                   "er bevorzugt Rotwein gegenüber Weißwein"),
    ("the cake was delicious",                                "der Kuchen war köstlich"),
    ("she is allergic to peanuts",                            "sie ist allergisch gegen Erdnüsse"),
    ("I drink two glasses of water every hour",               "ich trinke jede Stunde zwei Gläser Wasser"),
    ("the soup needs more salt",                              "die Suppe braucht mehr Salz"),
    ("he cooked a traditional German meal",                   "er hat ein traditionelles deutsches Gericht gekocht"),
    ("the apples are on the kitchen counter",                 "die Äpfel liegen auf der Küchentheke"),
    ("she baked a chocolate cake",                            "sie hat einen Schokoladenkuchen gebacken"),
    ("we had breakfast at the hotel",                         "wir haben im Hotel gefrühstückt"),
    ("the restaurant was fully booked",                       "das Restaurant war vollständig ausgebucht"),
    ("he ordered a beer at the bar",                          "er bestellte ein Bier an der Bar"),
    ("the vegetables are fresh from the market",              "das Gemüse ist frisch vom Markt"),
    ("she made a fruit salad for dessert",                    "sie hat einen Obstsalat als Nachspeise gemacht"),
    ("I like to eat cheese with bread",                       "ich esse gerne Käse mit Brot"),
    ("the steak was perfectly cooked",                        "das Steak war perfekt gegart"),
    ("we shared a bottle of champagne",                       "wir haben eine Flasche Champagner geteilt"),
    ("he eats breakfast at seven o clock",                    "er frühstückt um sieben Uhr"),
    ("the menu has many vegetarian options",                  "die Speisekarte hat viele vegetarische Optionen"),
    ("she drinks orange juice every morning",                 "sie trinkt jeden Morgen Orangensaft"),
    ("the ice cream melted quickly in the sun",               "das Eis schmolz schnell in der Sonne"),
    ("we ordered too much food",                              "wir haben zu viel Essen bestellt"),
    ("he is on a strict diet",                                "er macht eine strenge Diät"),
    ("the pasta was overcooked",                              "die Nudeln waren zu lange gekocht"),
    ("she added too much pepper to the sauce",                "sie hat zu viel Pfeffer in die Soße gegeben"),
    ("I need to buy sugar and flour",                         "ich muss Zucker und Mehl kaufen"),
    ("the fish was caught this morning",                      "der Fisch wurde heute Morgen gefangen"),
    ("he grilled sausages in the garden",                     "er hat Würstchen im Garten gegrillt"),
    ("the wine list was impressive",                          "die Weinkarte war beeindruckend"),
    ("she prepared a three course meal",                      "sie hat ein Drei-Gänge-Menü zubereitet"),
    ("I love eating spicy food",                              "ich esse gerne scharfes Essen"),
    ("the market sells fresh herbs and spices",               "der Markt verkauft frische Kräuter und Gewürze"),
    ("he drank too much coffee today",                        "er hat heute zu viel Kaffee getrunken"),
    ("the jam is homemade",                                   "die Marmelade ist hausgemacht"),
    ("she is learning to cook Italian food",                  "sie lernt italienisch zu kochen"),
    ("we went to a Chinese restaurant",                       "wir sind in ein chinesisches Restaurant gegangen"),
    ("the breakfast buffet opens at seven",                   "das Frühstücksbuffet öffnet um sieben Uhr"),
    ("he cut the bread into thin slices",                     "er hat das Brot in dünne Scheiben geschnitten"),
    ("the yoghurt is past its expiry date",                   "der Joghurt ist über sein Ablaufdatum"),
    ("she prepared a traditional recipe",                     "sie hat ein traditionelles Rezept zubereitet"),
    ("I prefer tea to coffee",                                "ich bevorzuge Tee gegenüber Kaffee"),
    ("the cheese has a strong flavour",                       "der Käse hat einen starken Geschmack"),
    ("he finished his meal quickly",                          "er hat seine Mahlzeit schnell beendet"),
    ("the waiter brought the wrong order",                    "der Kellner brachte die falsche Bestellung"),
    ("she is vegetarian and does not eat meat",               "sie ist Vegetarierin und isst kein Fleisch"),
    ("we shared the bill at the restaurant",                  "wir haben die Rechnung im Restaurant geteilt"),

    # ── 101-150  Travel and transportation ──────────────────────────────────
    ("the flight departs at noon",                            "der Flug startet um Mittag"),
    ("she lost her passport at the airport",                  "sie hat ihren Pass am Flughafen verloren"),
    ("we booked a hotel room for three nights",               "wir haben ein Hotelzimmer für drei Nächte gebucht"),
    ("the taxi driver knows the city well",                   "der Taxifahrer kennt die Stadt gut"),
    ("he bought a train ticket to Munich",                    "er hat eine Zugfahrkarte nach München gekauft"),
    ("the highway is closed due to an accident",              "die Autobahn ist wegen eines Unfalls gesperrt"),
    ("she arrived at the station ten minutes late",           "sie kam zehn Minuten zu spät am Bahnhof an"),
    ("we need to change trains at Frankfurt",                 "wir müssen in Frankfurt umsteigen"),
    ("the ferry crosses the river every hour",                "die Fähre überquert den Fluss jede Stunde"),
    ("I prefer to travel by train",                           "ich reise lieber mit dem Zug"),
    ("the plane landed safely at the airport",                "das Flugzeug landete sicher am Flughafen"),
    ("she rented a car for the weekend",                      "sie hat für das Wochenende ein Auto gemietet"),
    ("the tourist map shows all the landmarks",               "die Touristenkarte zeigt alle Sehenswürdigkeiten"),
    ("we visited the cathedral in the city centre",           "wir haben den Dom in der Stadtmitte besucht"),
    ("he took many photographs during the trip",              "er hat während der Reise viele Fotos gemacht"),
    ("the hotel has a swimming pool",                         "das Hotel hat ein Schwimmbad"),
    ("she asked the receptionist for directions",             "sie fragte die Empfangsdame nach dem Weg"),
    ("we checked in at the airport two hours early",          "wir haben zwei Stunden früh am Flughafen eingecheckt"),
    ("the luggage did not arrive with the flight",            "das Gepäck kam nicht mit dem Flug an"),
    ("I always get lost in big cities",                       "ich verirre mich immer in großen Städten"),
    ("the road is very narrow here",                          "die Straße ist hier sehr schmal"),
    ("she bought souvenirs for her family",                   "sie hat Souvenirs für ihre Familie gekauft"),
    ("we took the underground to the museum",                 "wir sind mit der U-Bahn zum Museum gefahren"),
    ("the bus was delayed by thirty minutes",                 "der Bus hatte dreißig Minuten Verspätung"),
    ("he had a window seat on the plane",                     "er hatte einen Fensterplatz im Flugzeug"),
    ("the campsite was next to a beautiful lake",             "der Campingplatz war neben einem wunderschönen See"),
    ("she visited seven countries in two weeks",              "sie hat in zwei Wochen sieben Länder besucht"),
    ("we drove through the mountains",                        "wir sind durch die Berge gefahren"),
    ("the traffic jam lasted for two hours",                  "der Stau dauerte zwei Stunden"),
    ("I love travelling in the summer",                       "ich reise gerne im Sommer"),
    ("the visa application takes several weeks",              "der Visumantrag dauert mehrere Wochen"),
    ("she always packs too many clothes",                     "sie packt immer zu viele Kleidungsstücke ein"),
    ("we stopped at a petrol station",                        "wir haben an einer Tankstelle angehalten"),
    ("the bridge was built in the nineteenth century",        "die Brücke wurde im neunzehnten Jahrhundert gebaut"),
    ("he missed the last train home",                         "er hat den letzten Zug nach Hause verpasst"),
    ("the airport is thirty kilometres from the city",        "der Flughafen ist dreißig Kilometer von der Stadt entfernt"),
    ("she swam in the sea every morning",                     "sie schwamm jeden Morgen im Meer"),
    ("we had a wonderful holiday in Italy",                   "wir hatten einen wunderbaren Urlaub in Italien"),
    ("the timetable shows the departure times",               "der Fahrplan zeigt die Abfahrtszeiten"),
    ("I need a window seat please",                           "ich brauche bitte einen Fensterplatz"),
    ("the cable car goes up to the summit",                   "die Seilbahn fährt bis zum Gipfel"),
    ("she explored the old town on foot",                     "sie erkundete die Altstadt zu Fuß"),
    ("we arrived safely after a long journey",                "wir sind nach einer langen Reise sicher angekommen"),
    ("the harbour was full of sailing boats",                 "der Hafen war voller Segelboote"),
    ("he drove on the wrong side of the road",                "er fuhr auf der falschen Seite der Straße"),
    ("the departure lounge was very crowded",                 "die Abflughalle war sehr voll"),
    ("she booked a window seat in advance",                   "sie hat im Voraus einen Fensterplatz gebucht"),
    ("we took a guided tour of the palace",                   "wir haben eine Führung durch den Palast gemacht"),
    ("the speed limit is one hundred here",                   "die Geschwindigkeitsbegrenzung beträgt hier hundert"),
    ("I always carry my passport when travelling",            "ich habe beim Reisen immer meinen Pass dabei"),

    # ── 151-200  Work and education ──────────────────────────────────────────
    ("she graduated from university last year",               "sie hat letztes Jahr ihr Studium abgeschlossen"),
    ("he is studying medicine at the university",             "er studiert Medizin an der Universität"),
    ("the meeting starts at ten o clock",                     "das Meeting beginnt um zehn Uhr"),
    ("we need to finish the project by Friday",               "wir müssen das Projekt bis Freitag abschließen"),
    ("she got a promotion at work",                           "sie hat eine Beförderung auf der Arbeit bekommen"),
    ("the office is on the fifth floor",                      "das Büro befindet sich im fünften Stock"),
    ("he sent the report by email",                           "er hat den Bericht per E-Mail gesendet"),
    ("the teacher explained the lesson clearly",              "die Lehrerin erklärte den Unterricht klar"),
    ("we have a presentation tomorrow morning",               "wir haben morgen früh eine Präsentation"),
    ("she is applying for a new job",                         "sie bewirbt sich um eine neue Stelle"),
    ("the school closes at three in the afternoon",           "die Schule schließt um drei Uhr nachmittags"),
    ("he studied hard for the mathematics exam",              "er hat hart für die Mathematikprüfung gelernt"),
    ("the deadline for the application is Monday",            "die Bewerbungsfrist ist Montag"),
    ("she works from home three days a week",                 "sie arbeitet drei Tage pro Woche von zu Hause"),
    ("the conference room is booked until noon",              "der Konferenzraum ist bis Mittag gebucht"),
    ("I am learning to speak German",                         "ich lerne Deutsch zu sprechen"),
    ("the professor gave an interesting lecture",             "der Professor hielt einen interessanten Vortrag"),
    ("he resigned from his position last month",              "er hat letzten Monat seinen Posten gekündigt"),
    ("the internship lasts three months",                     "das Praktikum dauert drei Monate"),
    ("she received a scholarship for her studies",            "sie erhielt ein Stipendium für ihr Studium"),
    ("the school trip is next Thursday",                      "der Schulausflug ist nächsten Donnerstag"),
    ("he is a very experienced engineer",                     "er ist ein sehr erfahrener Ingenieur"),
    ("the company has five hundred employees",                "das Unternehmen hat fünfhundert Mitarbeiter"),
    ("she teaches mathematics at the gymnasium",              "sie unterrichtet Mathematik am Gymnasium"),
    ("the annual report is ready for review",                 "der Jahresbericht ist zur Überprüfung fertig"),
    ("I finished my homework before dinner",                  "ich habe meine Hausaufgaben vor dem Abendessen fertiggemacht"),
    ("the business trip was very productive",                 "die Geschäftsreise war sehr produktiv"),
    ("he presented his research at the conference",           "er präsentierte seine Forschung auf der Konferenz"),
    ("the library has over one million books",                "die Bibliothek hat über eine Million Bücher"),
    ("she is training to become a nurse",                     "sie bildet sich zur Krankenschwester aus"),
    ("the exam results are published tomorrow",               "die Prüfungsergebnisse werden morgen veröffentlicht"),
    ("he takes the bus to work every day",                    "er fährt jeden Tag mit dem Bus zur Arbeit"),
    ("the new employee started on Monday",                    "der neue Mitarbeiter hat am Montag angefangen"),
    ("she wrote her thesis in six months",                    "sie hat ihre Abschlussarbeit in sechs Monaten geschrieben"),
    ("the students are preparing for exams",                  "die Studenten bereiten sich auf Prüfungen vor"),
    ("I work overtime almost every week",                     "ich arbeite fast jede Woche Überstunden"),
    ("the language course starts in September",               "der Sprachkurs beginnt im September"),
    ("he received a very good job offer",                     "er hat ein sehr gutes Stellenangebot erhalten"),
    ("the classroom has thirty students",                     "der Klassenraum hat dreißig Schüler"),
    ("she is the head of the marketing department",           "sie ist die Leiterin der Marketingabteilung"),
    ("the annual bonus was paid last week",                   "der Jahresbonus wurde letzte Woche ausgezahlt"),
    ("I need to prepare for the job interview",               "ich muss mich auf das Vorstellungsgespräch vorbereiten"),
    ("the school canteen has healthy food",                   "die Schulkantine hat gesundes Essen"),
    ("he has been working here for ten years",                "er arbeitet seit zehn Jahren hier"),
    ("the university library is open all night",              "die Universitätsbibliothek ist die ganze Nacht geöffnet"),
    ("she handed in her assignment on time",                  "sie hat ihre Aufgabe pünktlich abgegeben"),
    ("we need a bigger budget for this project",              "wir brauchen ein größeres Budget für dieses Projekt"),
    ("the new software makes work much easier",               "die neue Software macht die Arbeit viel einfacher"),
    ("he is the most talented student in class",              "er ist der talentierteste Schüler in der Klasse"),
    ("the company was founded in nineteen eighty",            "das Unternehmen wurde neunzehnhundertachtzig gegründet"),

    # ── 201-250  Family and relationships ────────────────────────────────────
    ("my grandmother makes the best apple cake",              "meine Großmutter macht den besten Apfelkuchen"),
    ("his sister lives in Hamburg",                           "seine Schwester lebt in Hamburg"),
    ("they got married in the summer",                        "sie haben im Sommer geheiratet"),
    ("my brother is two years older than me",                 "mein Bruder ist zwei Jahre älter als ich"),
    ("her parents live in the countryside",                   "ihre Eltern leben auf dem Land"),
    ("the family gathered for the holidays",                  "die Familie versammelte sich zu den Feiertagen"),
    ("he proposed to her on her birthday",                    "er hat ihr an ihrem Geburtstag einen Heiratsantrag gemacht"),
    ("my aunt and uncle have three children",                 "meine Tante und mein Onkel haben drei Kinder"),
    ("the baby took her first steps today",                   "das Baby hat heute seine ersten Schritte gemacht"),
    ("they celebrate Christmas together every year",          "sie feiern jedes Jahr gemeinsam Weihnachten"),
    ("my cousin is getting married in June",                  "mein Cousin heiratet im Juni"),
    ("she looks exactly like her mother",                     "sie sieht genau wie ihre Mutter aus"),
    ("the twins were born in January",                        "die Zwillinge wurden im Januar geboren"),
    ("he visited his grandfather every Sunday",               "er hat seinen Großvater jeden Sonntag besucht"),
    ("my parents met at university",                          "meine Eltern haben sich an der Universität kennengelernt"),
    ("she is expecting her second child",                     "sie erwartet ihr zweites Kind"),
    ("the family has a dog and two cats",                     "die Familie hat einen Hund und zwei Katzen"),
    ("his father retired last year",                          "sein Vater ist letztes Jahr in Rente gegangen"),
    ("my sister and I share a room",                          "meine Schwester und ich teilen uns ein Zimmer"),
    ("they have been together for five years",                "sie sind seit fünf Jahren zusammen"),
    ("the children visited their grandparents",               "die Kinder besuchten ihre Großeltern"),
    ("she named her daughter after her grandmother",          "sie benannte ihre Tochter nach ihrer Großmutter"),
    ("my niece is learning to ride a bicycle",                "meine Nichte lernt Fahrrad fahren"),
    ("they divorced after ten years of marriage",             "sie ließen sich nach zehn Jahren Ehe scheiden"),
    ("his brother moved to another country",                  "sein Bruder ist in ein anderes Land umgezogen"),
    ("my parents celebrated their anniversary",               "meine Eltern feierten ihren Hochzeitstag"),
    ("she calls her mother every Sunday",                     "sie ruft jeden Sonntag ihre Mutter an"),
    ("the family reunion was a great success",                "das Familientreffen war ein großer Erfolg"),
    ("he taught his son how to swim",                         "er brachte seinem Sohn das Schwimmen bei"),
    ("my grandfather served in the army",                     "mein Großvater diente in der Armee"),
    ("she grew up in a small village",                        "sie ist in einem kleinen Dorf aufgewachsen"),
    ("the children are very close to each other",             "die Kinder stehen sich sehr nahe"),
    ("his wife works as a doctor",                            "seine Frau arbeitet als Ärztin"),
    ("my youngest sister just started school",                "meine jüngste Schwester hat gerade mit der Schule angefangen"),
    ("they adopted a child from Romania",                     "sie haben ein Kind aus Rumänien adoptiert"),
    ("the family goes on holiday every summer",               "die Familie fährt jeden Sommer in den Urlaub"),
    ("he read his children a bedtime story",                  "er hat seinen Kindern eine Gutenachtgeschichte vorgelesen"),
    ("my mother is an excellent cook",                        "meine Mutter ist eine ausgezeichnete Köchin"),
    ("she missed her family while abroad",                    "sie hat ihre Familie im Ausland vermisst"),
    ("the new baby brought great joy to the family",          "das neue Baby brachte der Familie große Freude"),
    ("my father taught me how to drive",                      "mein Vater hat mir das Fahren beigebracht"),
    ("she is very proud of her children",                     "sie ist sehr stolz auf ihre Kinder"),
    ("they moved closer to be near the grandchildren",        "sie zogen näher heran um bei den Enkeln zu sein"),
    ("his parents speak four languages",                      "seine Eltern sprechen vier Sprachen"),
    ("my sister got a scholarship to study abroad",           "meine Schwester hat ein Stipendium für ein Auslandsstudium bekommen"),
    ("the family eats dinner together every evening",         "die Familie isst jeden Abend gemeinsam zu Abend"),
    ("she knitted a sweater for her grandson",                "sie hat einen Pullover für ihren Enkel gestrickt"),
    ("my brother plays football every weekend",               "mein Bruder spielt jedes Wochenende Fußball"),
    ("they renewed their wedding vows",                       "sie erneuerten ihre Eheversprechen"),
    ("my family is the most important thing to me",           "meine Familie ist mir das Wichtigste"),

    # ── 251-300  Nature and weather ──────────────────────────────────────────
    ("it is raining heavily outside",                         "es regnet draußen stark"),
    ("the mountains are covered in snow",                     "die Berge sind mit Schnee bedeckt"),
    ("the flowers bloom in spring",                           "die Blumen blühen im Frühling"),
    ("a strong wind is blowing today",                        "heute weht ein starker Wind"),
    ("the river flooded after the storm",                     "der Fluss ist nach dem Sturm überschwemmt"),
    ("the forest is full of wild animals",                    "der Wald ist voller wilder Tiere"),
    ("the sunset was absolutely beautiful",                   "der Sonnenuntergang war absolut wunderschön"),
    ("there was a thick fog this morning",                    "heute Morgen gab es einen dichten Nebel"),
    ("the lake is frozen in winter",                          "der See ist im Winter zugefroren"),
    ("the birds sing early in the morning",                   "die Vögel singen früh am Morgen"),
    ("the harvest was very good this year",                   "die Ernte war dieses Jahr sehr gut"),
    ("a rainbow appeared after the rain",                     "nach dem Regen erschien ein Regenbogen"),
    ("the temperature dropped below zero tonight",            "die Temperatur fiel heute Nacht unter null"),
    ("the leaves are turning yellow and red",                 "die Blätter werden gelb und rot"),
    ("there was a thunderstorm last night",                   "letzte Nacht gab es ein Gewitter"),
    ("the garden needs watering every day",                   "der Garten muss jeden Tag gegossen werden"),
    ("the volcano erupted last century",                      "der Vulkan brach letztes Jahrhundert aus"),
    ("the ocean is very deep at this point",                  "der Ozean ist an dieser Stelle sehr tief"),
    ("the desert receives very little rainfall",              "die Wüste bekommt sehr wenig Niederschlag"),
    ("the full moon was visible last night",                  "der Vollmond war letzte Nacht sichtbar"),
    ("the earthquake caused serious damage",                  "das Erdbeben verursachte schwere Schäden"),
    ("butterflies are attracted to bright colours",           "Schmetterlinge werden von hellen Farben angezogen"),
    ("the snow melted quickly in the afternoon",              "der Schnee ist am Nachmittag schnell geschmolzen"),
    ("the forest fire destroyed thousands of trees",          "der Waldbrand zerstörte tausende Bäume"),
    ("the sun set behind the mountains",                      "die Sonne versank hinter den Bergen"),
    ("heavy rain is forecast for the weekend",                "für das Wochenende ist starker Regen vorhergesagt"),
    ("the wild horses ran across the plains",                 "die Wildpferde liefen über die Ebenen"),
    ("the river flows towards the sea",                       "der Fluss fließt in Richtung Meer"),
    ("the polar ice caps are melting",                        "die polaren Eiskappen schmelzen"),
    ("the cherry trees are in full blossom",                  "die Kirschbäume stehen in voller Blüte"),
    ("the climate has changed significantly",                 "das Klima hat sich erheblich verändert"),
    ("a flock of birds flew over the village",                "ein Vogelschwarm flog über das Dorf"),
    ("the waterfall was a spectacular sight",                 "der Wasserfall war ein spektakulärer Anblick"),
    ("the grass is greener after the rain",                   "das Gras ist nach dem Regen grüner"),
    ("the spider built a beautiful web",                      "die Spinne hat ein wunderschönes Netz gebaut"),
    ("the night sky was full of stars",                       "der Nachthimmel war voller Sterne"),
    ("the coral reef is home to many fish",                   "das Korallenriff ist die Heimat vieler Fische"),
    ("autumn is my favourite season",                         "der Herbst ist meine Lieblingszeit"),
    ("the oak tree is over three hundred years old",          "die Eiche ist über dreihundert Jahre alt"),
    ("the bee collects nectar from the flowers",              "die Biene sammelt Nektar von den Blumen"),
    ("the storm uprooted several large trees",                "der Sturm hat mehrere große Bäume entwurzelt"),
    ("the valley is surrounded by high mountains",            "das Tal ist von hohen Bergen umgeben"),
    ("snow fell throughout the night",                        "es schneite die ganze Nacht"),
    ("the spring water is very clean and cold",               "das Quellwasser ist sehr sauber und kalt"),
    ("the cat caught a mouse in the garden",                  "die Katze hat eine Maus im Garten gefangen"),
    ("the pond is full of frogs in summer",                   "der Teich ist im Sommer voller Frösche"),
    ("it was the hottest summer on record",                   "es war der heißeste Sommer seit Beginn der Aufzeichnungen"),
    ("the fox ran into the forest",                           "der Fuchs lief in den Wald"),
    ("the clouds blocked the sunlight",                       "die Wolken blockierten das Sonnenlicht"),
    ("the swallow returns every spring",                      "die Schwalbe kehrt jeden Frühling zurück"),

    # ── 301-350  Shopping and money ──────────────────────────────────────────
    ("the shoes are on sale this week",                       "die Schuhe sind diese Woche im Angebot"),
    ("she spent too much money on clothes",                   "sie hat zu viel Geld für Kleidung ausgegeben"),
    ("the price has increased significantly",                 "der Preis ist erheblich gestiegen"),
    ("I need to withdraw money from the bank",                "ich muss Geld von der Bank abheben"),
    ("the shopping centre opens at nine",                     "das Einkaufszentrum öffnet um neun Uhr"),
    ("he paid for the groceries with cash",                   "er hat die Lebensmittel mit Bargeld bezahlt"),
    ("the receipt shows the total amount",                    "der Kassenbon zeigt den Gesamtbetrag"),
    ("she returned the dress because it was too small",       "sie hat das Kleid zurückgegeben weil es zu klein war"),
    ("the bookshop has a large selection",                    "die Buchhandlung hat eine große Auswahl"),
    ("I am saving money for a new computer",                  "ich spare Geld für einen neuen Computer"),
    ("the discount applies to all products",                  "der Rabatt gilt für alle Produkte"),
    ("she bought a birthday present for her friend",          "sie hat ein Geburtstagsgeschenk für ihre Freundin gekauft"),
    ("the exchange rate is very favourable today",            "der Wechselkurs ist heute sehr günstig"),
    ("he spent his entire salary in one week",                "er hat sein gesamtes Gehalt in einer Woche ausgegeben"),
    ("the online store delivers within two days",             "der Online-Shop liefert innerhalb von zwei Tagen"),
    ("she haggled over the price at the market",              "sie hat auf dem Markt um den Preis gefeilscht"),
    ("the credit card payment was declined",                  "die Kreditkartenzahlung wurde abgelehnt"),
    ("I always compare prices before buying",                 "ich vergleiche immer die Preise vor dem Kauf"),
    ("the winter sale starts after Christmas",                "der Winterschlussverkauf beginnt nach Weihnachten"),
    ("he found a very good deal online",                      "er hat online ein sehr gutes Angebot gefunden"),
    ("the jewellery shop was very expensive",                 "das Juweliergeschäft war sehr teuer"),
    ("she ordered new furniture for the living room",         "sie hat neue Möbel für das Wohnzimmer bestellt"),
    ("the queue at the checkout was very long",               "die Schlange an der Kasse war sehr lang"),
    ("I need to pay the electricity bill",                    "ich muss die Stromrechnung bezahlen"),
    ("the antique shop had some interesting items",           "das Antiquitätengeschäft hatte einige interessante Gegenstände"),
    ("he lost his wallet on the bus",                         "er hat sein Portemonnaie im Bus verloren"),
    ("the bakery sells bread at half price in the evening",   "die Bäckerei verkauft Brot am Abend zum halben Preis"),
    ("she received a gift voucher for her birthday",          "sie hat einen Geschenkgutschein zu ihrem Geburtstag bekommen"),
    ("the bank charges a fee for this service",               "die Bank erhebt eine Gebühr für diesen Service"),
    ("I prefer to shop in small local stores",                "ich kaufe lieber in kleinen lokalen Geschäften ein"),
    ("the toy store is very popular with children",           "das Spielzeuggeschäft ist bei Kindern sehr beliebt"),
    ("she tried on three different dresses",                  "sie hat drei verschiedene Kleider anprobiert"),
    ("the price tag shows thirty euros",                      "das Preisschild zeigt dreißig Euro"),
    ("he ordered a new laptop online",                        "er hat einen neuen Laptop online bestellt"),
    ("the market is held every Saturday morning",             "der Markt findet jeden Samstagmorgen statt"),
    ("she always buys organic vegetables",                    "sie kauft immer Bio-Gemüse"),
    ("the parking fee is two euros per hour",                 "die Parkgebühr beträgt zwei Euro pro Stunde"),
    ("I need to return this item",                            "ich muss diesen Artikel zurückgeben"),
    ("the sales assistant was very helpful",                  "die Verkäuferin war sehr hilfsbereit"),
    ("he bought the last pair of shoes in the store",         "er hat das letzte Schuhpaar im Laden gekauft"),
    ("the tax is included in the price",                      "die Steuer ist im Preis inbegriffen"),
    ("she found her dream dress at the boutique",             "sie hat ihr Traumkleid in der Boutique gefunden"),
    ("the supermarket was out of milk",                       "der Supermarkt hatte keine Milch mehr"),
    ("I spent three hours shopping today",                    "ich habe heute drei Stunden eingekauft"),
    ("the new collection arrives next month",                 "die neue Kollektion kommt nächsten Monat"),
    ("he donated old clothes to the charity shop",            "er hat alte Kleidung an den Second-Hand-Laden gespendet"),
    ("the cost of living has risen sharply",                  "die Lebenshaltungskosten sind stark gestiegen"),
    ("she bought a new handbag in the sale",                  "sie hat in der Sale-Aktion eine neue Handtasche gekauft"),
    ("the gift was beautifully wrapped",                      "das Geschenk war wunderschön eingepackt"),
    ("I always check the expiry date",                        "ich überprüfe immer das Ablaufdatum"),

    # ── 351-400  Health and body ─────────────────────────────────────────────
    ("she has an appointment with the doctor",                "sie hat einen Termin beim Arzt"),
    ("he broke his arm playing football",                     "er hat sich beim Fußballspielen den Arm gebrochen"),
    ("the headache lasted for several hours",                 "der Kopfschmerz dauerte mehrere Stunden"),
    ("I need to take my medicine twice a day",                "ich muss meine Medizin zweimal täglich nehmen"),
    ("the hospital is five minutes from here",                "das Krankenhaus ist fünf Minuten von hier entfernt"),
    ("she recovered quickly from her illness",                "sie erholte sich schnell von ihrer Krankheit"),
    ("he runs five kilometres every morning",                 "er läuft jeden Morgen fünf Kilometer"),
    ("the dentist said I need a filling",                     "der Zahnarzt sagte dass ich eine Füllung brauche"),
    ("she is taking vitamins to stay healthy",                "sie nimmt Vitamine um gesund zu bleiben"),
    ("the blood test results came back normal",               "die Bluttest-Ergebnisse kamen normal zurück"),
    ("he pulled a muscle during training",                    "er hat beim Training einen Muskel gezerrt"),
    ("I have been feeling tired all week",                    "ich habe mich die ganze Woche müde gefühlt"),
    ("the ambulance arrived within ten minutes",              "der Krankenwagen kam innerhalb von zehn Minuten"),
    ("she is allergic to penicillin",                         "sie ist allergisch gegen Penicillin"),
    ("the operation was completely successful",               "die Operation war vollständig erfolgreich"),
    ("he has worn glasses since the age of twelve",           "er trägt seit dem Alter von zwölf Jahren eine Brille"),
    ("the physiotherapist recommended daily exercises",       "der Physiotherapeut empfahl tägliche Übungen"),
    ("I caught a cold after getting wet in the rain",         "ich habe mich erkältet nachdem ich im Regen nass wurde"),
    ("she lost ten kilograms through diet and exercise",      "sie hat durch Diät und Sport zehn Kilo abgenommen"),
    ("the prescription can be collected at the pharmacy",     "das Rezept kann in der Apotheke abgeholt werden"),
    ("he had a knee replacement operation",                   "er hatte eine Knieersatzoperation"),
    ("she meditates for twenty minutes each morning",         "sie meditiert jeden Morgen zwanzig Minuten"),
    ("the nurse checked his blood pressure",                  "die Krankenschwester maß seinen Blutdruck"),
    ("I feel much better after sleeping well",                "ich fühle mich nach einem guten Schlaf viel besser"),
    ("the doctor prescribed antibiotics",                     "der Arzt verschrieb Antibiotika"),
    ("she goes to the gym three times a week",                "sie geht dreimal pro Woche ins Fitnessstudio"),
    ("the child has a high fever",                            "das Kind hat hohes Fieber"),
    ("he suffered a heart attack last year",                  "er erlitt letztes Jahr einen Herzanfall"),
    ("the surgeon performed a difficult operation",           "der Chirurg führte eine schwierige Operation durch"),
    ("I need to wear sunscreen in summer",                    "ich muss im Sommer Sonnencreme verwenden"),
    ("she twisted her ankle on the stairs",                   "sie hat sich auf der Treppe den Knöchel verstaucht"),
    ("the vaccine protects against the virus",                "der Impfstoff schützt gegen das Virus"),
    ("he suffers from back pain regularly",                   "er leidet regelmäßig unter Rückenschmerzen"),
    ("the health insurance covers the costs",                 "die Krankenversicherung übernimmt die Kosten"),
    ("she drinks enough water every day",                     "sie trinkt jeden Tag genug Wasser"),
    ("the x-ray showed no broken bones",                      "das Röntgenbild zeigte keine gebrochenen Knochen"),
    ("I have been sneezing all morning",                      "ich niese den ganzen Morgen"),
    ("the dietitian recommended a balanced diet",             "die Ernährungsberaterin empfahl eine ausgewogene Ernährung"),
    ("he had an eye test at the optician",                    "er machte beim Optiker einen Sehtest"),
    ("the waiting room was full of patients",                 "das Wartezimmer war voller Patienten"),
    ("she is training for a half marathon",                   "sie trainiert für einen Halbmarathon"),
    ("the cast on his leg was removed today",                 "der Gips an seinem Bein wurde heute entfernt"),
    ("I need to eat more fruit and vegetables",               "ich muss mehr Obst und Gemüse essen"),
    ("the pharmacy is open twenty four hours",                "die Apotheke ist vierundzwanzig Stunden geöffnet"),
    ("she finally quit smoking last year",                    "sie hat letztes Jahr endlich mit dem Rauchen aufgehört"),
    ("the doctor recommended plenty of rest",                 "der Arzt empfahl viel Ruhe"),
    ("he has been suffering from insomnia",                   "er leidet unter Schlaflosigkeit"),
    ("the scan revealed a small cyst",                        "der Scan enthüllte eine kleine Zyste"),
    ("I feel dizzy when I stand up too quickly",              "mir wird schwindelig wenn ich zu schnell aufstehe"),
    ("she has regular check-ups with her doctor",             "sie hat regelmäßige Vorsorgeuntersuchungen bei ihrem Arzt"),

    # ── 401-450  Time, numbers and descriptions ──────────────────────────────
    ("there are twenty four hours in a day",                  "ein Tag hat vierundzwanzig Stunden"),
    ("she arrived five minutes before the meeting",           "sie kam fünf Minuten vor dem Meeting an"),
    ("the building is forty metres tall",                     "das Gebäude ist vierzig Meter hoch"),
    ("it takes about an hour to get there",                   "es dauert etwa eine Stunde um dorthin zu kommen"),
    ("the concert lasted three hours",                        "das Konzert dauerte drei Stunden"),
    ("he is thirty five years old",                           "er ist fünfunddreißig Jahre alt"),
    ("the room is twelve metres long",                        "das Zimmer ist zwölf Meter lang"),
    ("she woke up at half past six",                          "sie ist um halb sieben aufgewacht"),
    ("there are seven days in a week",                        "eine Woche hat sieben Tage"),
    ("the new building cost five million euros",              "das neue Gebäude kostete fünf Millionen Euro"),
    ("she has been living here for six years",                "sie wohnt seit sechs Jahren hier"),
    ("the meeting lasted longer than expected",               "das Meeting dauerte länger als erwartet"),
    ("he is the tallest person in the group",                 "er ist die größte Person in der Gruppe"),
    ("the painting is very old and very valuable",            "das Gemälde ist sehr alt und sehr wertvoll"),
    ("the sky turned completely dark at eight",               "der Himmel wurde um acht Uhr komplett dunkel"),
    ("she is a very kind and generous person",                "sie ist eine sehr freundliche und großzügige Person"),
    ("the road is about fifty kilometres long",               "die Straße ist etwa fünfzig Kilometer lang"),
    ("he answered the question correctly",                    "er hat die Frage richtig beantwortet"),
    ("the first floor is above the ground floor",             "das erste Stockwerk ist über dem Erdgeschoss"),
    ("it was the longest day of the year",                    "es war der längste Tag des Jahres"),
    ("she smiled warmly at the guests",                       "sie lächelte die Gäste herzlich an"),
    ("there are twelve months in a year",                     "ein Jahr hat zwölf Monate"),
    ("the package weighs three kilograms",                    "das Paket wiegt drei Kilogramm"),
    ("he spoke very quietly in the library",                  "er sprach sehr leise in der Bibliothek"),
    ("the project was completed ahead of schedule",           "das Projekt wurde vor dem Zeitplan abgeschlossen"),
    ("she painted the walls light blue",                      "sie hat die Wände hellblau gestrichen"),
    ("the temperature is minus ten degrees",                  "die Temperatur beträgt minus zehn Grad"),
    ("he answered all fifty questions correctly",             "er hat alle fünfzig Fragen richtig beantwortet"),
    ("the journey takes approximately two hours",             "die Reise dauert ungefähr zwei Stunden"),
    ("she is the youngest of four siblings",                  "sie ist die jüngste von vier Geschwistern"),
    ("the old bridge was built in twelve hundred",            "die alte Brücke wurde im Jahr zwölfhundert gebaut"),
    ("he spoke three languages before he was ten",            "er sprach drei Sprachen bevor er zehn war"),
    ("the apartment has four rooms and a balcony",            "die Wohnung hat vier Zimmer und einen Balkon"),
    ("she sings beautifully and plays the piano",             "sie singt wunderschön und spielt Klavier"),
    ("the film is two hours and twenty minutes long",         "der Film ist zwei Stunden und zwanzig Minuten lang"),
    ("he arrived exactly on time",                            "er kam genau pünktlich an"),
    ("the box is too heavy to carry alone",                   "die Schachtel ist zu schwer um sie alleine zu tragen"),
    ("she has a very good memory",                            "sie hat ein sehr gutes Gedächtnis"),
    ("the door is two metres high",                           "die Tür ist zwei Meter hoch"),
    ("he is the most experienced person here",                "er ist die erfahrenste Person hier"),
    ("the restaurant is very popular on weekends",            "das Restaurant ist am Wochenende sehr beliebt"),
    ("she learned to speak English in two years",             "sie hat in zwei Jahren Englisch sprechen gelernt"),
    ("the match ended in a draw",                             "das Spiel endete unentschieden"),
    ("he has lived in Germany for fifteen years",             "er wohnt seit fünfzehn Jahren in Deutschland"),
    ("the clock on the wall shows three o clock",             "die Uhr an der Wand zeigt drei Uhr"),
    ("she sent over a hundred emails yesterday",              "sie hat gestern über hundert E-Mails gesendet"),
    ("the hall can seat five hundred people",                 "der Saal bietet fünfhundert Personen Platz"),
    ("it is already the end of the month",                    "es ist schon das Ende des Monats"),
    ("he has read over three hundred books",                  "er hat über dreihundert Bücher gelesen"),
    ("the train was exactly two minutes late",                "der Zug hatte genau zwei Minuten Verspätung"),

    # ── 451-500  Emotions, questions and miscellaneous ───────────────────────
    ("she was very happy to see her friends again",           "sie war sehr glücklich ihre Freunde wiederzusehen"),
    ("he felt nervous before the presentation",               "er fühlte sich vor der Präsentation nervös"),
    ("I am sorry for being late",                             "es tut mir leid dass ich zu spät bin"),
    ("she cried when she heard the sad news",                 "sie weinte als sie die traurigen Nachrichten hörte"),
    ("where is the nearest train station please",             "wo ist bitte der nächste Bahnhof"),
    ("do you speak English",                                  "sprechen Sie Englisch"),
    ("I do not understand what you are saying",               "ich verstehe nicht was Sie sagen"),
    ("could you please speak more slowly",                    "könnten Sie bitte langsamer sprechen"),
    ("what time does the next bus leave",                     "wann fährt der nächste Bus ab"),
    ("how much does this cost",                               "wie viel kostet das"),
    ("he was surprised by the unexpected gift",               "er war von dem unerwarteten Geschenk überrascht"),
    ("she felt proud after winning the award",                "sie fühlte sich stolz nachdem sie die Auszeichnung gewonnen hatte"),
    ("I am very excited about the trip",                      "ich freue mich sehr auf die Reise"),
    ("can you help me please",                                "können Sie mir bitte helfen"),
    ("where can I find a good restaurant",                    "wo finde ich ein gutes Restaurant"),
    ("he was angry about the mistake",                        "er war über den Fehler verärgert"),
    ("she felt relieved after the exam",                      "sie fühlte sich nach der Prüfung erleichtert"),
    ("I am afraid of flying",                                 "ich habe Angst vor dem Fliegen"),
    ("what is your name please",                              "wie heißen Sie bitte"),
    ("he regretted not studying harder",                      "er bedauerte es nicht härter gelernt zu haben"),
    ("she laughed at the funny joke",                         "sie lachte über den lustigen Witz"),
    ("I miss my family very much",                            "ich vermisse meine Familie sehr"),
    ("could I have the bill please",                          "könnte ich bitte die Rechnung haben"),
    ("he was embarrassed by the situation",                   "er war von der Situation peinlich berührt"),
    ("she felt lonely in the big city",                       "sie fühlte sich in der großen Stadt einsam"),
    ("I am grateful for your help",                           "ich bin dankbar für Ihre Hilfe"),
    ("the children were excited about Christmas",             "die Kinder freuten sich auf Weihnachten"),
    ("he was disappointed with the result",                   "er war mit dem Ergebnis enttäuscht"),
    ("is there a pharmacy near here",                         "gibt es eine Apotheke in der Nähe"),
    ("she felt confident during the interview",               "sie fühlte sich während des Vorstellungsgesprächs selbstbewusst"),
    ("I apologise for the misunderstanding",                  "ich entschuldige mich für das Missverständnis"),
    ("he was delighted to receive the award",                 "er war begeistert die Auszeichnung zu erhalten"),
    ("can you recommend a good hotel",                        "können Sie ein gutes Hotel empfehlen"),
    ("she was shocked by the unexpected news",                "sie war von den unerwarteten Nachrichten schockiert"),
    ("I need to find the city centre",                        "ich muss das Stadtzentrum finden"),
    ("he felt hopeful about the future",                      "er fühlte sich hoffnungsvoll über die Zukunft"),
    ("she was bored during the long meeting",                 "sie war während des langen Meetings gelangweilt"),
    ("how far is it to the airport",                          "wie weit ist es bis zum Flughafen"),
    ("I am very pleased to meet you",                         "es freut mich sehr Sie kennenzulernen"),
    ("he was curious about the new technology",               "er war neugierig auf die neue Technologie"),
    ("she felt calm and relaxed at the spa",                  "sie fühlte sich im Spa ruhig und entspannt"),
    ("what language do they speak in Austria",                "welche Sprache spricht man in Österreich"),
    ("I am looking forward to the weekend",                   "ich freue mich auf das Wochenende"),
    ("he smiled when he saw the results",                     "er lächelte als er die Ergebnisse sah"),
    ("she was moved by the beautiful music",                  "sie war von der schönen Musik bewegt"),
    ("I feel very comfortable in this city",                  "ich fühle mich in dieser Stadt sehr wohl"),
    ("he was tired after the long journey",                   "er war nach der langen Reise müde"),
    ("she cheered loudly at the football match",              "sie jubelte laut beim Fußballspiel"),
    ("thank you very much for your kindness",                 "vielen Dank für Ihre Freundlichkeit"),
    ("I hope to see you again soon",                          "ich hoffe Sie bald wiederzusehen"),
]

# 80 / 20 train–validation split.
split_idx = int(len(RAW_CORPUS) * 0.80)
# Avoid landing on a round-number boundary (e.g. exactly 400).
# The SageMaker Seq2Seq container uses 0-indexed record counting; when the
# file contains exactly N records it attempts to read record #N, gets EOF,
# and reports a phantom "empty sentence" which propagates NaN into the
# mean target/source length ratio and causes a fatal type-conversion error.
if split_idx % 100 == 0:
    split_idx -= 1                     # use 399 instead of 400

TRAIN_PAIRS = RAW_CORPUS[:split_idx]   # 399 pairs
VAL_PAIRS   = RAW_CORPUS[split_idx:]   # 101 pairs

print("=" * 70)
print("  seq-to-seq-translation-en-de.py")
print("  English → German Machine Translation — SageMaker Seq2Seq")
print("=" * 70)
print(f"\n  Corpus : {len(RAW_CORPUS)} sentence pairs  "
      f"({len(TRAIN_PAIRS)} train / {len(VAL_PAIRS)} val)")


# ===========================================================================
# STEP 1 — TOKENISATION
# ===========================================================================
# Tokenisation converts a raw sentence string into a list of word tokens.
#
# This example uses a simple lowercase + punctuation-splitting approach.
# For production workloads consider:
#   SentencePiece (BPE or unigram subword) — handles morphology and OOV well
#   spaCy                                  — language-aware, handles contractions
#   Moses tokenizer — standard WMT pre-processing tool
# ===========================================================================

def tokenise(sentence: str) -> list:
    """
    Split a sentence into lowercase word tokens.

    Punctuation attached to words is separated by inserting spaces, so
    "mat."  becomes  ["mat", "."].

    Parameters
    ----------
    sentence : Raw input sentence string.

    Returns
    -------
    List of lowercase word token strings.

    Examples
    --------
    >>> tokenise("The cat sat on the mat.")
    ['the', 'cat', 'sat', 'on', 'the', 'mat', '.']
    >>> tokenise("She speaks three languages fluently!")
    ['she', 'speaks', 'three', 'languages', 'fluently', '!']
    """
    sentence = sentence.lower()
    sentence = re.sub(r"([.,!?;:\"'()\[\]{}])", r" \1 ", sentence)
    return [t for t in sentence.split() if t]


# Tokenise every sentence pair in the full corpus.
tokenised_pairs = [(tokenise(en), tokenise(de)) for en, de in RAW_CORPUS]
print(f"\n  Sample tokenisation (pair 0):")
en_toks_ex, de_toks_ex = tokenised_pairs[0]
print(f"    English : {en_toks_ex}")
print(f"    German  : {de_toks_ex}")


# ===========================================================================
# STEP 2 — BUILD VOCABULARIES
# ===========================================================================
# A vocabulary maps each unique word to a unique integer token ID.
#
# Token ID layout
# ---------------
#   0          <pad>   zero-padding for sequences shorter than the maximum
#   1          <eos>   end-of-sequence marker appended to every target
#   2          <unk>   replaces any word absent from the vocabulary
#   3 … N      real words sorted by descending frequency
#              (most common word receives the lowest available ID)
#
# Two separate vocabularies are required
# --------------------------------------
#   vocab.src.json  — English word → integer ID
#   vocab.trg.json  — German  word → integer ID
#
# Both files are uploaded to the same "vocab" S3 channel prefix.
# The container reads them by exact file name; any other name is ignored.
# ===========================================================================

def build_vocab(token_lists: list, max_size: int) -> dict:
    """
    Build a word → integer_token_id vocabulary from a list of token lists.

    Parameters
    ----------
    token_lists : List of tokenised sentences (each a list of str).
    max_size    : Maximum vocabulary size including the three special tokens.
                  Words beyond this limit are handled with <unk> at encode time.

    Returns
    -------
    vocab : dict  {word_string: integer_token_id}
            e.g.  {"<pad>": 0, "<eos>": 1, "<unk>": 2, "the": 3, "a": 4, ...}
    """
    # Count word frequencies across all sentences.
    counter = collections.Counter()
    for tokens in token_lists:
        counter.update(tokens)

    # Keep only the (max_size - WORD_OFFSET) most frequent words so there is
    # room for the three special tokens at IDs 0, 1, and 2.
    top_words = [word for word, _ in counter.most_common(max_size - WORD_OFFSET)]

    # Assign integer IDs — special tokens first, then real words in
    # descending frequency order starting at WORD_OFFSET (= 3).
    vocab = {
        "<pad>": PAD_ID,   # 0
        "<eos>": EOS_ID,   # 1
        "<unk>": UNK_ID,   # 2
    }
    for rank, word in enumerate(top_words):
        vocab[word] = rank + WORD_OFFSET   # 3, 4, 5, …

    return vocab


print("\n" + "=" * 70)
print("STEP 2 — Building vocabularies")
print("=" * 70)

# Build from the full corpus so validation words are also covered.
all_en_tokens = [en for en, _ in tokenised_pairs]
all_de_tokens = [de for _, de in tokenised_pairs]

src_vocab = build_vocab(all_en_tokens, MAX_VOCAB_SIZE)
trg_vocab = build_vocab(all_de_tokens, MAX_VOCAB_SIZE)

# Write vocab files with the exact names the container expects.
with open(SRC_VOCAB_FILE, "w", encoding="utf-8") as f:
    json.dump(src_vocab, f, indent=2, ensure_ascii=False)

with open(TRG_VOCAB_FILE, "w", encoding="utf-8") as f:
    json.dump(trg_vocab, f, indent=2, ensure_ascii=False)

src_vocab_size = len(src_vocab)
trg_vocab_size = len(trg_vocab)

print(f"  English vocab : {src_vocab_size:,} tokens  →  {SRC_VOCAB_FILE}")
print(f"  German  vocab : {trg_vocab_size:,} tokens  →  {TRG_VOCAB_FILE}")

print(f"\n  First 8 entries — English vocab (vocab.src.json):")
for word, tok_id in list(src_vocab.items())[:8]:
    print(f"    {repr(word):<12}  →  {tok_id}")

print(f"\n  First 8 entries — German vocab (vocab.trg.json):")
for word, tok_id in list(trg_vocab.items())[:8]:
    print(f"    {repr(word):<12}  →  {tok_id}")

print(f"""
  Vocab file structure (vocab.src.json excerpt):
  ───────────────────────────────────────────────
  {{
    "<pad>": 0,    ← padding token
    "<eos>": 1,    ← end-of-sequence
    "<unk>": 2,    ← unknown / out-of-vocabulary
    "the":   3,    ← most frequent word
    "a":     4,
    "is":    5,
    ...
  }}
""")


# ===========================================================================
# STEP 3 — ENCODE SENTENCES AS TOKEN ID SEQUENCES
# ===========================================================================
# Each sentence is converted from a list of word strings to a fixed-length
# list of integer token IDs.
#
# Source encoding (English)
#   1. Look up each word in src_vocab; use UNK_ID for unknown words.
#   2. Truncate to MAX_SEQ_LEN_SOURCE if longer.
#   3. Right-pad with PAD_ID (0) to reach exactly MAX_SEQ_LEN_SOURCE.
#   Note: the encoder does not need an explicit <eos> — the container adds it.
#
# Target encoding (German)
#   1. Look up each word in trg_vocab; use UNK_ID for unknown words.
#   2. Append EOS_ID (1) to mark the end of the sentence.
#   3. Truncate to MAX_SEQ_LEN_TARGET (the <eos> counts toward the limit).
#   4. Right-pad with PAD_ID (0) to reach exactly MAX_SEQ_LEN_TARGET.
# ===========================================================================

def encode_source(tokens: list, vocab: dict, max_len: int) -> list:
    """
    Convert source (English) tokens to a right-padded integer ID sequence.

    Parameters
    ----------
    tokens  : Word token list from tokenise().
    vocab   : English vocabulary dict {word: token_id}.
    max_len : Output sequence length (MAX_SEQ_LEN_SOURCE).

    Returns
    -------
    List of exactly max_len integers.

    Example
    -------
    tokens = ['the', 'cat', 'sat', 'on', 'the', 'mat']
    → IDs    : [3, 12, 87, 9, 3, 63]
    → padded to 50: [3, 12, 87, 9, 3, 63, 0, 0, ..., 0]
    """
    ids = [vocab.get(tok, UNK_ID) for tok in tokens]
    ids = ids[:max_len]
    ids += [PAD_ID] * (max_len - len(ids))
    return ids


def encode_target(tokens: list, vocab: dict, max_len: int) -> list:
    """
    Convert target (German) tokens to a right-padded integer ID sequence
    with <eos> appended before padding.

    Parameters
    ----------
    tokens  : Word token list from tokenise().
    vocab   : German vocabulary dict {word: token_id}.
    max_len : Output sequence length (MAX_SEQ_LEN_TARGET).

    Returns
    -------
    List of exactly max_len integers.

    Example
    -------
    tokens = ['die', 'Katze', 'saß', 'auf', 'der', 'Matte']
    → IDs         : [7, 18, 243, 14, 22, 115]
    → append <eos>: [7, 18, 243, 14, 22, 115, 1]
    → padded to 50: [7, 18, 243, 14, 22, 115, 1, 0, ..., 0]
    """
    ids = [vocab.get(tok, UNK_ID) for tok in tokens]
    ids.append(EOS_ID)
    ids = ids[:max_len]
    ids += [PAD_ID] * (max_len - len(ids))
    return ids


# Demonstrate encoding on the first sentence pair.
en_toks, de_toks = tokenised_pairs[0]
sample_src_ids = encode_source(en_toks, src_vocab, MAX_SEQ_LEN_SOURCE)
sample_tgt_ids = encode_target(de_toks, trg_vocab, MAX_SEQ_LEN_TARGET)

print("=" * 70)
print("STEP 3 — Encoding sentences to token ID sequences")
print("=" * 70)
print(f"\n  Sentence pair 0:")
print(f"    English : {en_toks}")
print(f"    German  : {de_toks}")
print(f"\n  source_ids (first 10 of {MAX_SEQ_LEN_SOURCE}):")
print(f"    {sample_src_ids[:10]} ...")
print(f"\n  target_ids (first 10 of {MAX_SEQ_LEN_TARGET}):")
print(f"    {sample_tgt_ids[:10]} ...")
print(f"    (EOS_ID={EOS_ID} follows the word IDs; remaining positions are PAD_ID=0)")


# ===========================================================================
# STEP 4 — WRITE RECORDIO PROTOBUF FILES
# ===========================================================================
# Each .rec file holds a sequence of RecordIO frames.  Every frame encodes
# one sentence pair using two nested layers:
#
#   Inner layer — SageMaker protobuf Record
#       record.features["source_ids"].int32_tensor  ←  English token IDs
#       record.features["target_ids"].int32_tensor  ←  German  token IDs
#
#   Outer layer — RecordIO frame (little-endian)
#       Bytes 0–3 :  0xCED7230A  magic number  (uint32)
#       Bytes 4–7 :  payload length in bytes   (uint32)
#       Bytes 8–N :  serialised protobuf bytes
#       Padding   :  zero bytes to align the next frame to a 4-byte boundary
# ===========================================================================

def build_proto_record(source_ids: list, target_ids: list) -> bytes:
    """
    Serialise one (source_ids, target_ids) pair as a SageMaker protobuf Record.

    Parameters
    ----------
    source_ids : Encoded English token ID list of length MAX_SEQ_LEN_SOURCE.
    target_ids : Encoded German  token ID list of length MAX_SEQ_LEN_TARGET.

    Returns
    -------
    Serialised protobuf bytes ready to be wrapped in a RecordIO frame.
    """
    record = Record()
    record.features["source_ids"].int32_tensor.values.extend(source_ids)
    record.features["target_ids"].int32_tensor.values.extend(target_ids)
    return record.SerializeToString()


def write_recordio_file(path: str, pairs: list) -> int:
    """
    Encode all sentence pairs and write them to a binary RecordIO file.

    Parameters
    ----------
    path  : Output file path (.rec).
    pairs : List of (english_tokens, german_tokens) tuples.

    Returns
    -------
    Total bytes written to the file.
    """
    total_bytes = 0
    skipped = 0
    with open(path, "wb") as f:
        for en_toks, de_toks in pairs:
            # ── Guard 1: skip pairs where tokenisation returned nothing ────
            if not en_toks or not de_toks:
                skipped += 1
                continue

            src_ids = encode_source(en_toks, src_vocab, MAX_SEQ_LEN_SOURCE)
            tgt_ids = encode_target(de_toks, trg_vocab, MAX_SEQ_LEN_TARGET)

            # ── Guard 2: skip all-padding source sequences ────────────────
            # An all-PAD source (source_len = 0) produces target/source = NaN,
            # which the container cannot convert to an integer and raises a
            # fatal "Customer Error: cannot convert float NaN to integer".
            if all(i == PAD_ID for i in src_ids):
                skipped += 1
                continue

            proto_bytes = build_proto_record(src_ids, tgt_ids)
            length      = len(proto_bytes)

            # 8-byte frame header: magic number + payload length (little-endian).
            f.write(struct.pack("<II", RECORDIO_MAGIC, length))
            f.write(proto_bytes)

            # Zero-pad so the next frame starts on a 4-byte boundary.
            pad_len = (4 - length % 4) % 4
            if pad_len:
                f.write(b"\x00" * pad_len)

            total_bytes += 8 + length + pad_len

    if skipped:
        print(f"  WARNING: Skipped {skipped} empty / all-padding sentence pair(s)")
    return total_bytes


print("\n" + "=" * 70)
print("STEP 4 — Writing RecordIO protobuf files")
print("=" * 70)

train_tok = [(tokenise(en), tokenise(de)) for en, de in TRAIN_PAIRS]
val_tok   = [(tokenise(en), tokenise(de)) for en, de in VAL_PAIRS]

t_bytes = write_recordio_file(TRAIN_FILE, train_tok)
v_bytes = write_recordio_file(VAL_FILE,   val_tok)

print(f"  {TRAIN_FILE} : {len(train_tok)} records  ({t_bytes:,} bytes)")
print(f"  {VAL_FILE}   : {len(val_tok)} records  ({v_bytes:,} bytes)")

# Read back the first frame and decode it to verify the round-trip.
print(f"\n  Verifying frame 0 of {TRAIN_FILE}:")
with open(TRAIN_FILE, "rb") as f:
    _, length = struct.unpack("<II", f.read(8))
    payload   = f.read(length)

probe = Record()
probe.ParseFromString(payload)
src_ids = list(probe.features["source_ids"].int32_tensor.values)
tgt_ids = list(probe.features["target_ids"].int32_tensor.values)

print(f"    source_ids (first 10) : {src_ids[:10]} ...")
print(f"    target_ids (first 10) : {tgt_ids[:10]} ...")

id_to_src = {v: k for k, v in src_vocab.items()}
id_to_trg = {v: k for k, v in trg_vocab.items()}

decoded_en = [id_to_src.get(i, "<unk>") for i in src_ids if i != PAD_ID]
decoded_de = [id_to_trg.get(i, "<unk>") for i in tgt_ids if i not in (PAD_ID, EOS_ID)]

print(f"    Decoded English : {decoded_en}")
print(f"    Decoded German  : {decoded_de}")


# ===========================================================================
# STEP 5 — UPLOAD DATA AND VOCAB FILES TO S3
# ===========================================================================
# The Seq2Seq container reads from three named S3 data channels:
#
#   "train"
#       Binary RecordIO file with training sentence pairs.
#       Content-type: application/x-recordio-protobuf
#
#   "validation"
#       Binary RecordIO file with validation sentence pairs.
#       The container evaluates BLEU score on this set after every checkpoint.
#       Content-type: application/x-recordio-protobuf
#
#   "vocab"
#       S3 prefix that contains vocab.src.json and vocab.trg.json.
#       The container mounts this channel at /opt/ml/input/data/vocab/.
#       Content-type: application/json
#
# The vocab files are uploaded with boto3.upload_file() to keep the S3 key
# names exactly as vocab.src.json and vocab.trg.json, which the container
# requires.  Using session.upload_data() would add a filename prefix.
# ===========================================================================
print("\n" + "=" * 70)
print("STEP 5 — Uploading data and vocab files to S3")
print("=" * 70)

boto3_session = boto3.Session(region_name=REGION)
sm_session    = Session(boto_session=boto3_session)
s3_client     = boto3.client("s3", region_name=REGION)

print("  Uploading translation-train.rec ...")
s3_train_uri = sm_session.upload_data(
    path=TRAIN_FILE,
    bucket=S3_BUCKET,
    key_prefix=f"{S3_PREFIX}/input/train",
)
print(f"    → {s3_train_uri}")

print("  Uploading translation-val.rec ...")
s3_val_uri = sm_session.upload_data(
    path=VAL_FILE,
    bucket=S3_BUCKET,
    key_prefix=f"{S3_PREFIX}/input/validation",
)
print(f"    → {s3_val_uri}")

print("  Uploading vocab.src.json ...")
s3_client.upload_file(
    Filename=SRC_VOCAB_FILE,
    Bucket=S3_BUCKET,
    Key=f"{S3_PREFIX}/input/vocab/vocab.src.json",
)
print(f"    → s3://{S3_BUCKET}/{S3_PREFIX}/input/vocab/vocab.src.json")

print("  Uploading vocab.trg.json ...")
s3_client.upload_file(
    Filename=TRG_VOCAB_FILE,
    Bucket=S3_BUCKET,
    Key=f"{S3_PREFIX}/input/vocab/vocab.trg.json",
)
print(f"    → s3://{S3_BUCKET}/{S3_PREFIX}/input/vocab/vocab.trg.json")


# ===========================================================================
# SNS NOTIFICATION HELPERS
# ===========================================================================
# These helpers publish a single SNS message when the training job finishes.
# They are no-ops when SNS_TOPIC_ARN is None.
# ===========================================================================

def _sns_notify(subject: str, message: str) -> None:
    """
    Publish *message* to SNS_TOPIC_ARN with the given *subject*.

    Parameters
    ----------
    subject : Short one-line subject shown in e-mail / SMS notifications.
    message : Full notification body (plain text).
    """
    if not SNS_TOPIC_ARN:
        return
    try:
        sns_client = boto3.client("sns", region_name=REGION)
        sns_client.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=subject,
            Message=message,
        )
        print(f"  [SNS] Notification sent  →  {SNS_TOPIC_ARN}")
    except Exception as sns_err:          # never let a notification crash the script
        print(f"  [SNS] WARNING: could not publish notification: {sns_err}")


def _notify_training_success(job_name: str, model_artefact: str) -> None:
    """Send a success notification after a completed training job."""
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    _sns_notify(
        subject=f"[SUCCESS] SageMaker training job completed — {job_name}",
        message=(
            f"Training job  : {job_name}\n"
            f"Status        : COMPLETED\n"
            f"Finished at   : {ts}\n"
            f"Model artefact: {model_artefact}\n"
        ),
    )


def _notify_training_failure(job_name: str, error: Exception) -> None:
    """Send a failure notification when a training job raises an exception."""
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    _sns_notify(
        subject=f"[FAILED] SageMaker training job failed — {job_name}",
        message=(
            f"Training job  : {job_name}\n"
            f"Status        : FAILED\n"
            f"Failed at     : {ts}\n"
            f"Error         : {error}\n"
        ),
    )


# ===========================================================================
# STEP 6 — CONFIGURE AND LAUNCH THE TRAINING JOB
# ===========================================================================
# Hyperparameter reference:
# https://docs.aws.amazon.com/sagemaker/latest/dg/seq-2-seq-hyperparameters.html
#
# vocab_size_source / vocab_size_target
#   Must be set explicitly because source and target have different vocabulary
#   sizes.  Without these the container infers sizes from the maximum token ID
#   in the .rec file, which can be smaller than the true vocabulary and cause
#   out-of-bounds embedding access.
#
# rnn_attention_type = "mlp"
#   Multi-layer perceptron (Bahdanau) attention is the standard choice for
#   natural language translation.
#
# optimized_metric = "bleu"
#   BLEU (Bilingual Evaluation Understudy) measures n-gram overlap between
#   model output and the reference translation.
#   Score range: 0 (no overlap) to 100 (perfect match).
#   A score above 20 is generally considered usable for a translation model.
# ===========================================================================
print("\n" + "=" * 70)
print("STEP 6 — Configuring and launching the training job")
print("=" * 70)

seq2seq_image_uri = get_image_uri(
    region=REGION,
    framework="seq2seq",
    version="1",
)
print(f"  Container image : {seq2seq_image_uri}")

hyperparameters = {

    # Sequence lengths — must match the encoding step.
    "max_seq_len_source": MAX_SEQ_LEN_SOURCE,
    "max_seq_len_target": MAX_SEQ_LEN_TARGET,

    # Token embeddings.
    "num_embed_source": 512,                # encoder token embedding size
    "num_embed_target": 512,                # decoder token embedding size

    # Encoder / decoder architecture.
    "encoder_type":       "rnn",            # rnn (LSTM/GRU) or cnn
    "decoder_type":       "rnn",            # rnn (LSTM/GRU) or cnn
    "num_layers_encoder": 1,                # stacked rnn layers in encoder
    "num_layers_decoder": 1,                # stacked rnn layers in decoder

    # RNN cell settings.
    "rnn_num_hidden":         512,          # hidden units — must be even (biLSTM)
    "rnn_cell_type":          "lstm",       # lstm or gru
    "rnn_attention_type":     "dot",        # dot | fixed | mlp | bilinear
    "rnn_decoder_state_init": "last",       # seed decoder with encoder's final state

    # Optimiser.
    "max_num_epochs":   10,                 # stop after 10 full passes
    "learning_rate":    0.0003,             # Adam initial learning rate
    "optimizer_type":   "adam",             # adam | sgd | rmsprop
    "weight_init_type": "xavier",           # uniform or xavier

    # Evaluation and early stopping.
    "optimized_metric":                 "bleu",   # perplexity | accuracy | bleu
    # With ~399 training records and default batch_size ≈ 64 there are only
    # ~6 batches/epoch × 10 epochs ≈ 60 total batches.  Setting this to 1000
    # means a checkpoint is never reached, early-stopping never fires, and
    # internal statistics relying on checkpointing can get stuck.  Use a
    # value smaller than the total expected batch count.
    "checkpoint_frequency_num_batches": 10,       # evaluate every ~2 epochs
    "checkpoint_threshold":             3,        # stop if no improvement for 3 checkpoints
}

print("\n  Hyperparameters:")
for k, v in hyperparameters.items():
    print(f"    {k:<35} : {v}")

estimator = Estimator(
    image_uri=seq2seq_image_uri,
    role=ROLE_ARN,
    instance_count=INSTANCE_COUNT,
    instance_type=TRAINING_INSTANCE_TYPE,
    volume_size=30,
    output_path=f"s3://{S3_BUCKET}/{S3_PREFIX}/output",
    max_run=7200,
    hyperparameters=hyperparameters,
    # Encrypts all inter-node traffic with TLS/SSL when instance_count > 1.
    # Has no effect on single-instance jobs but is kept as a security best
    # practice so encryption is automatically applied if INSTANCE_COUNT is
    # increased to run distributed training in the future.
    encrypt_inter_container_traffic=True,
    sagemaker_session=sm_session,
)

train_input = TrainingInput(
    s3_data=f"s3://{S3_BUCKET}/{S3_PREFIX}/input/train",
    content_type="application/x-recordio-protobuf",
)
validation_input = TrainingInput(
    s3_data=s3_val_uri,
    content_type="application/x-recordio-protobuf",
)
vocab_input = TrainingInput(
    # Points to the S3 prefix containing vocab.src.json and vocab.trg.json.
    # The container mounts this channel at /opt/ml/input/data/vocab/.
    # Omitting this channel causes a "Vocab files not present" error.
    s3_data=f"s3://{S3_BUCKET}/{S3_PREFIX}/input/vocab",
    content_type="application/json",
)

print("\n  Starting training job  (monitor in SageMaker console or CloudWatch) ...")
_training_job_name = estimator.latest_training_job.name if estimator.latest_training_job else "seq2seq-en-de"
try:
    estimator.fit(
        inputs={
            "train":      train_input,
            "validation": validation_input,
            "vocab":      vocab_input,
        },
        wait=True,
        logs=True,
    )
    # Refresh the job name after fit() has created the job.
    _training_job_name = estimator.latest_training_job.name
    print(f"\n  Training complete.")
    print(f"  Model artefact : {estimator.model_data}")
    _notify_training_success(_training_job_name, estimator.model_data)
except Exception as _training_exc:
    # Attempt to capture the job name even on failure.
    try:
        _training_job_name = estimator.latest_training_job.name
    except Exception:
        pass
    _notify_training_failure(_training_job_name, _training_exc)
    raise


# ===========================================================================
# STEP 7 — DEPLOY ENDPOINT AND RUN INFERENCE
# ===========================================================================
# Request format sent to the endpoint:
#   {"instances": [{"source": [word_id_1, word_id_2, ..., word_id_50]}]}
#
# Response format returned by the endpoint:
#   {"predictions": [{"score": [...], "target": [word_id_1, ..., word_id_50]}]}
#
# Predicted token IDs are decoded back to German words using id_to_trg.
# PAD_ID (0) and EOS_ID (1) are stripped before displaying the result.
# ===========================================================================
print("\n" + "=" * 70)
print("STEP 7 — Deploying endpoint and running inference")
print("=" * 70)

predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type=INFERENCE_INSTANCE_TYPE,
    endpoint_name="seq2seq-translation-en-de",
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer(),
)
print(f"  Endpoint 'seq2seq-translation-en-de' is InService.")

# Use the first validation pair as the inference demo.
demo_en_tokens, demo_de_tokens = val_tok[0]
demo_src_ids    = encode_source(demo_en_tokens, src_vocab, MAX_SEQ_LEN_SOURCE)
demo_tgt_ids_gt = encode_target(demo_de_tokens, trg_vocab, MAX_SEQ_LEN_TARGET)

print(f"\n  Input  (English)      : {demo_en_tokens}")
print(f"  Source IDs (first 10) : {demo_src_ids[:10]} ...")

response = predictor.predict(
    {"instances": [{"source": demo_src_ids}]}
)

# Decode predicted token IDs → German words, stripping PAD and EOS.
predicted_ids   = response["predictions"][0]["target"]
predicted_words = [
    id_to_trg.get(i, "<unk>")
    for i in predicted_ids
    if i not in (PAD_ID, EOS_ID)
]

# Decode ground-truth target for comparison.
gt_words = [
    id_to_trg.get(i, "<unk>")
    for i in demo_tgt_ids_gt
    if i not in (PAD_ID, EOS_ID)
]

print(f"\n  Reference (German)  : {gt_words}")
print(f"  Predicted (German)  : {predicted_words}")

# Delete the endpoint when finished to avoid ongoing charges.
predictor.delete_endpoint()
print(f"\n  Endpoint deleted.  Done.")

