def get_test_queries(lang):
    if lang == "eng":
        queries = {
            # "factual": [
            #     "List some courses related to embedded systems.",
            #     "Which courses cover machine learning or artificial intelligence?",
            #     "Are there any courses focused on computer networks or networking?",
            #     "Name all courses that involve programming in C or C++.",
            #     "Which courses are related to software engineering or software architecture?",
            #     "List courses that include database topics."
            # ],
            # "comparison": [
            #     "What is the difference between Machine Learning and Deep Learning courses — do they cover similar topics?",
            #     "How does Artificial Intelligence differ from Computer Vision in terms of content and goals?",
            #     "Are Computer Vision and image processing separate courses, and if so, what distinguishes them?",
            #     "Do courses about embedded systems and microcontroller programming overlaping in topics, or do they cover distinct material?",
            #     "Are there any software architecture courses and how do they differ from software engineering courses?"
            # ],
            "timetable": [
                "When are the lab sessions for Embedded Systems?",
                "When does the Machine Learning course have lectures?",
                "How many hours per week of studz does the Linear Algebra course require?",
                "Do the lab sessions for Intelligent systems and Machine Perception overlap in the schedule?",
                "Can you give me a list of courses in the first year of the university program?",
                "Are there any courses with lab sessions on Fridays?"
            ],
            "complex": [
                "Can a student take both Electronic Business and Compilers in the same semester, or do they conflict?",
                "Is Mathematics a prerequisite for any other course?",
                "Which courses from the second year are recommended before enrolling in Machine perception?"
            ]
        }
    elif lang == "slo":
        queries = {
            # "factual": [
            #     "Ali mi lahko poves par predmetov vezanih na vgrajene sisteme?",
            #     "Kateri predmeti pokrivajo strojno učenje ali umetno inteligenco?",
            #     "Ali obstajajo predmeti, osredotočeni na računalniška omrežja?",
            #     "Naštej vse predmete, ki vključujejo programiranje v C ali C++.",
            #     "Kateri predmeti so povezani s programskim inženirstvom ali programsko arhitekturo?",
            #     "Naštej predmete, ki vključujejo tematiko podatkovnih baz."
            # ],
            # "comparison": [
            #     "Kakšna je razlika med predmetoma Strojno učenje in Globoko učenje — ali pokrivata podobne teme?",
            #     "Kako se umetna inteligenca razlikuje od računalniškega vida glede vsebine in ciljev?",
            #     "Ali sta Računalniški vid in obdelava slik ločena predmeta in po čem se razlikujeta?",
            #     "Ali se predmeti za vgrajene sisteme in predmeti za programiranje mikrokrmilnikov vsebinsko prekrivata ali pokrivata različno snov?",
            #     "Ali obstajajo predmeti za softversko arhitekturo in kako se razlikujejo od softverskega inzenirstva?"
            # ],
            # "timetable": [
            #     "Kdaj so vaje pri predmetu Vgrajeni sistemi?",
            #     "Kdaj ima predmet Strojno učenje predavanja?",
            #     "Koliko ur tedenskog učenja zahteva predmet Linearna algebra?",
            #     "Ali se vaje pri predmetih Inteligentni sistemi in Umetno zaznavanje časovno prekrivata?",
            #     "Ali obstajajo predmeti, ki imajo vaje v petek?"
            # ],
            "complex": [
                # "Ali lahko študent izbere Elektronsko poslovanje in Prevajalnike v istem semestru ali se urnika prekrivata?",
                # "Ali je matematika predpogoj za kateri drug predmet?",
                "Katere predmete drugega letnika je priporočljivo opraviti pred vpisom v predmet Umetno zaznavanje?"
            ]
        }

    return queries

def get_single_query():
    query = {
        "single": [
            "Naštej vse predmete, ki vključujejo programiranje v C ali C++."
        ]
    }

    return query