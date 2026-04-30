"""
Script d'organisation de la Knowledge Base ML/DS
=================================================
Déplace les PDFs en 3 dossiers principaux avec sous-dossiers :

  📁 a_supprimer/
      ├── 01_superficiel_grand_public/
      ├── 02_oriente_business/
      ├── 03_hardware_embedded/
      ├── 04_trop_specialise_niche/
      └── 05_specifique_R/

  📁 redondants/
      ├── 01_doublons_editions/
      ├── 02_meme_auteur_contenu_proche/
      └── 03_genai_en_exces/

  📁 a_persister/
      ├── 01_fondements_ML_theorie_algorithmes/
      ├── 02_deep_learning/
      ├── 03_NLP/
      ├── 04_data_science_appliquee_python/
      ├── 05_mathematiques_pour_ML/
      ├── 06_IA_generale_introductions/
      ├── 07_feature_engineering_modelisation/
      └── 08_sujets_complementaires/

Usage :
    python organise_books.py --source /chemin/vers/pdfs
    python organise_books.py --source /chemin/vers/pdfs --dry-run
"""

import shutil
import argparse
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
#  CLASSIFICATION COMPLÈTE
# ═══════════════════════════════════════════════════════════════

STRUCTURE = {

    # ── A_SUPPRIMER ────────────────────────────────────────────
    "a_supprimer": {

        "01_superficiel_grand_public": [
            "Machine Learning for Absolute Beginners - Oliver Theobald.pdf",
            "Machine Learning for dummies (john paul mueller).pdf",
            "Machine Learning_ An Essential Guide to Machine Learning for Beginners - Herbert Jones.pdf",
            "Machine Learning_ Step-by-Step Guide To Implement Machine Learning Algorithms with Python - Rudolph Russell.pdf",
            "learning to love data science - Mike Barlow.PDF",
            "Predictive Analytics - THE POWER TO PREDICT WHO WILL CLICK, BUY, LIE, OR DIE - ERIC SIEGEL.pdf",
            "Machine Learning Crash Course for Engineers (Eklas Hossain).pdf",
        ],

        "02_oriente_business": [
            "Artificial Intelligence and Machine Learning for Business - A No-Nonsense Guide to Data Driven Technologies - Steven Finlay .pdf",
            "Artificial Intelligence for Business -  What You Need to Know about Machine Learning and Neural Networks -  Doug Rose.pdf",
            "Machine Learning for Decision Makers -Cognitive Computing Fundamentals for Better Decision Making - Patanjali Kashyap.pdf",
            "Data Science for Business - foster provost & tom fawcett.pdf",
            "Data Science - JOHN D. KELLEHER , BRENDAN TIERNEY.pdf",
        ],

        "03_hardware_embedded": [
            # "Embedded Artificial Intelligence - Bin Li.pdf",
            # "Embedded Deep Learning Algorithms, Architectures and Circuits for Always-on Neural Network Processing (Bert Moons + Daniel Bankman + Marian Verhelst).pdf",
        ],

        "04_trop_specialise_niche": [
            "Gaussian Processes for Machine Learning - Carl Edward Rasmussen  , Christopher K. I. Williams.pdf",
            "Statistical Reinforcement Learning_ Modern Machine Learning Approaches - Masashi Sugiyama.pdf",
            "Introduction to Statistical Relational Learning - LISE GETOOR AND BEN TASKAR .pdf",
            "Analysis and Comparison of Metaheuristics -volume  1063  - Erik Cuevas + Omar Avalos + Jorge Galvez.pdf",
            "Applications of Linear and Nonlinear Models(Erik W. Grafarend + autre).pdf",
            "Machine Learning for Audio, Image and Video Analysis - Francesco Camastra • Alessandro Vinciarelli.pdf",
            "MATLAB Machine Learning Recipes - 3rd Edition - (Michael Paluszek  ).pdf",
            "Distributed Machine Learning with PySpark - Abdelaziz Testas.pdf",
            "Data Fabric and Data Mesh Approaches with AI( Eberhard Hechler + ).pdf",
            "Numerical algorithms - methods for computer vision, machine learning, and graphics - Justin Solomon.pdf",
            "DEVELOPING KAGGLE NOTEBOOKS - Gabriel Preda.pdf",
            "Bad Data Handbook - Q. Ethan McCallum.pdf",
        ],

        "05_specifique_R": [
            "Beginning Data Science in R(Thomas Mailund).pdf",
            "Introduction to Deep Learning Using R_ A Step-by-Step Guide to Learning and Implementing Deep Learning Models Using R - Taweh Beysolow.pdf",
            "Data Insight Foundations Step-by-Step Data Analysis with R (Nikita Tkachenko).pdf",
        ],
    },

    # ── REDONDANTS ─────────────────────────────────────────────
    "redondants": {

        "01_doublons_editions": [
            # 1ères éditions à supprimer — garder la 2e édition de chaque
            # "Introduction to Artificial Intelligence - Wolfgang Ertel.pdf", 
            # "Machine Learning with TensorFlow - Nishat Shukla.pdf",
            # "Practical Machine Learning with Python - Dipanjan Sarkar , Raghav Bali , Tushar Sharma.pdf",
            # "Basics for Linear Algebra for Machine Learning - Discover the Mathematical Language of Data in Python - Jason Brownlee.pdf",
        ],
        # je ne suis pas conviencu que les 4 titres suivants soient vraiment redondants, mais ils sont très proches de livres déjà conservés (même auteur ou contenu très similaire) et je pense que ça ne nuit pas à la diversité de garder une seule version de chaque thème traité par Jason Brownlee et Pramod Singh

        "02_meme_auteur_contenu_proche": [
            # Même auteur, contenu très proche d'un autre livre conservé
            "Master Machine Learning Algorithms - Discover how they work - Json Brownlee.pdf",
            "Advanced Data Analytics Using Python - Sayan Mukhopadhyay.pdf",
            "Machine Learning with PySpark - With Natural Language Processing and Recommender Systems - Pramod Singh.pdf",
        ],

        "03_genai_en_exces": [
            # 4 livres GenAI superficiels en trop
            # Garder : Applications of Generative AI + Understanding Generative AI Business Applications
            "Applied Generative AI for Beginners (Akshay Kulkarni + autre).pdf",
            "Building Generative AI-Powered Apps(Arushi Kansal).pdf",
            "Generative AI (How ChatGPT and Other AI Tools) (Tom Taulli).pdf",
            "Generative Artificial Intelligence (Drupad K Khublani and  Shivam R Solanki).pdf",
        ],
    },

    # ── A_PERSISTER ────────────────────────────────────────────
    "a_persister": {

        "01_fondements_ML_theorie_algorithmes": [
            "A First Course in Machine Learning – Simon Rogers Mark Girolami.pdf",
            "Foundations of Machine Learning - Mehryar Mohri, Afshin Rostamizadeh, and Ameet Talwalkar.pdf",
            "Machine Learning - An Algorithmic Perspective - 2nd Edition - Stepen Marsland .pdf",
            "Machine Learning Refined Foundations, Algorithms, and Applications - Jeremy Watt • Reza Borhani • Aggelos K. Katsaggelos.pdf",
            "Machine Learning(Zhi-Hua Zhou).pdf",
            "Principles And Theory For Data Mining And Machine Learning - Bertrand Clarke · Ernest Fokoue´ · Hao Helen Zhang.pdf",
            "Machine Learning and Data Science - Danierl D. Gustierres.pdf",
            "Introduction to Machine Learning with Python - A GUIDE FOR DATA SCIENTISTS - Andreas C. Müller & Sarah Guido.pdf",
            "Machine learning in action - Peter Harrington - MANNING.pdf",
            "Machine Learning in Python - Michael Bowles.pdf",
            "Pro Machine Learning Algorithms - - V Kishore Ayyadevara.pdf",
            "Machine Learning Hands-On for Developers and Technical Professionals - Json Bell.pdf",
            "Machine Learning Mastery with Python - Understand Your Data, Create Accurate Models and Work Projects End-To-End - Jason Brownlee.pdf",
            "From Curve Fitting to Machine Learning - 2end edition - Achim Zielesny .pdf",
        ],

        "02_deep_learning": [
            "Deep Learning - Ian Goodfellow ,Yoshua Bengio ,Aaron Courville.pdf",
            "Deep Learning with Python - SECOND EDITION - François Chollet.pdf",
            "Deep Learning with PyTorch -Eli Stevens , Luca Antiga.pdf",
            "Neural Networks and Deep Learning -  A Textbook - Charu C. Aggarwal.pdf",
            "Deep Learning Architectures(Ovidiu Calin).pdf",
            "Foundations of Deep Learning(Fengxiang He and Dacheng Tao).pdf",
            "Advances in Deep Learning -Volume 2 -M Arif Wani + Bisma Sultan + Sarwat Ali + Mukhtar Ahmed Sofi.pdf",
            "Applied Deep Learning - Umberto Michelucci.pdf",
            "Modern Deep Learning for Tabular Data (Andre Ye and Zian Wang).pdf",
            "Deep Learning for Natural Language Processing Creating Neural Networks with Python ( Palash goyal + sumit pandey + karan jain).pdf",
            "Deep Learning with Applications Using Python (Chatbots and Face, Object, and Speech ) (Navin Kumar Manaswi + ).pdf",
            "PyTorch Recipes(Pradeepta Mishra).pdf",
        ],

        "03_NLP": [
            "Handbook of Natural Language Processing - 2nd Edition - Ralf  Herbich , Thore Graepel.pdf",
            "Applied Natural Language Processing with Python - Taweh Beysolow.pdf",
            "Applied Text Analysis with Python - Enabling Language Aware Data Products with Machine Learning  - Benjamin Bengfort, Rebecca Bilbro, and Tony Ojeda.pdf",
            "Natural Language Processing Recipes_ Unlocking Text Data with Machine Learning and Deep Learning using Python - Akshay Kulkarni , Adarsha Shivananda.pdf",
            "Natural Language Processing- 2nd Edition - (Raymond Lee).pdf",
            "Building Chatbots with Python - Using Natural Language Processing and Machine Learning - Sumit Raj.pdf",
            "The Practical Guide to Large Language Models - Hands-On AI Applications with Hugging Face Transformers - Ivan Gridin.pdf",
        ],

        "04_data_science_appliquee_python": [
            "Hands On Machine Learning with Scikit-Learn Keras and TensorFlow - 2nd Edition - Aurelien Geron.pdf",
            "Data Science from Scratch - FIRST PRINCIPLES WITH PYTHON - Joel Grus.pdf",
            "Data Science Essentials in Python - Collect , Organize , Explore , Predict ,Value - Dmitry Zinoviev.pdf",
            "Data Science Fundamentals for Python and MongoDB(David Paper).pdf",
            "Mastering Machine Learning with Python in Six Steps_ A Practical Implementation Guide to Predictive Data Analytics Using Python - Manohar Swamynathan.pdf",
            "Mastering Machine Learning with scikit-learn-  2nd edition - Gavin Hackeling.pdf",
            "Practical Machine Learning with Python  A Problem-Solver's - Guide to Building Real-World Intelligent Systems - Dipanjan Sarkar Raghav Bali Tushar Sharma.pdf",
            "Python Machine Learning - - Wei-Meng Lee.pdf",
            "Real-World Machine Learning - Henrik Brink.pdf",
            "Data Mining - Practical Machine Learning Tools and Techniques - Ian H. Witten , Eibe Frank , Mark A. Hall.pdf",
            "Data Wrangling with Python - Tips and Tools to Make Your Life Easier - Jacqueline Kazil & Katharine Jarmul.pdf",
        ],

        "05_mathematiques_pour_ML": [
            "Algebra, Topology, Differential Calculus, and Optimization Theory For Computer Science and Machine Learning - Jean Gallier and Jocelyn Quaintance.pdf",
            "Applied Linear Algebra (Peter J Olver  Chehrzad Shakiban).pdf",
            "Basics of Linear Algebra for Machine Learning - Jason Brownlee.pdf",
            "Introduction to Probability - Joseph K. Blitzstein, Jessica Hwang.pdf",
            "Probability and Statistics for Machine Learning - Text Book -   (Charu C. Aggarwa).pdf",
            "Practical Statistics For Data Scientists Practical Statistics For Data Scientists - Peter Bruce , Andrew Bruce , Peter Gedeck.pdf",
            "An Introduction to Statistical Methods and Data Analysis - 7th Edition -  R Lyman Ott + Michael Longnecker.pdf",
        ],

        "06_IA_generale_introductions": [
            "Introduction To Artificial intelligence - 2nd Edition - (Wolfgang Ertel).pdf",
            "Artificial Intelligence - With an Introduction to Machine Learning - Richard E. Neapolitan and Xia Jiang.pdf",
            "Artificial Intelligence for Engineers(Zhen Leo Liu).pdf",
            "Introduction to Pattern Recognition and Machine Learning - Murty & Devi .pdf",
            "Introducing Data Science_ Big Data, Machine Learning, and more, using Python tools - DAVY CIELEN , ARNO D. B. MEYSMAN , MOHAMED ALI.pdf",
            "Introduction To Data Science - Laura Igual  Santi Seguí.pdf",
            "The Data Science Design Manual -Steven S.Skiena.pdf",
        ],

        "07_feature_engineering_modelisation": [
            "Feature Engineering for Machine Learning -  Principles and Techniques for Data Scientists - Alice Zheng & Amanda Casari.pdf",
            "Explainable and Interpretable Models in Computer Vision and Machine Learning - Hugo Jair Escalante • Sergio Escalera.pdf",
            "Handbook of Statistics  Machine Learning Volume 31- Venu Govindaraju.pdf",
            "Handbook of Statistics volume 24- Data Mining and Data Visualization- CR Rao.pdf",
            "Principles of Data Science -  SHAUN V. AULT , SOOHYUN NAM LIAO.pdf",
        ],

        "08_sujets_complementaires": [
            "The Art of Reinforcement Learning(Michael Hu).pdf",
            "Genetic Algorithms and Machine Learning for Programmers_ Create AI Models and Evolve Solutions - Frances Buontempo.pdf",
            "Metaheuristics for Machine Learning(Mansour Eddaly Bassem Jarboui ).pdf",
            "Practical Machine Learning and Image Processing_ For Facial Recognition, Object Detection, and Pattern Recognition - Himanshu Singh.pdf",
            "Applications of Generative AI(Zhihan Lyu).pdf",
            "Understanding Generative AI Business Applications - A Guide to Technical Principles and Real-World Applications -  (Irena Cronin).pdf",
            "Machine Learning with TensorFlow - 2nd edition -Nishant Shukla .pdf",
        ],
    },
}

# ═══════════════════════════════════════════════════════════════
#  UTILITAIRES
# ═══════════════════════════════════════════════════════════════

def normalise(name: str) -> str:
    """Normalise un nom de fichier pour comparaison insensible à la casse."""
    return name.strip().lower()

def find_file(source_dir: Path, target_name: str) -> Path | None:
    """Cherche un fichier dans source_dir : exact d'abord, normalisé ensuite."""
    exact = source_dir / target_name
    if exact.exists():
        return exact
    target_norm = normalise(target_name)
    for f in source_dir.iterdir():
        if f.is_file() and normalise(f.name) == target_norm:
            return f
    return None

def move_file(src: Path, dst_dir: Path, dry_run: bool) -> bool:
    dst = dst_dir / src.name
    if dry_run:
        print(f"      → {src.name}")
        return True
    try:
        shutil.move(str(src), str(dst))
        return True
    except Exception as e:
        print(f"      [ERREUR] {src.name} : {e}")
        return False

# ═══════════════════════════════════════════════════════════════
#  ORGANISATION PRINCIPALE
# ═══════════════════════════════════════════════════════════════

ICONES = {
    "a_supprimer": "❌",
    "redondants":  "⚠️ ",
    "a_persister": "✅",
}

def organise(source_dir: Path, dry_run: bool = False):
    if not source_dir.exists():
        print(f"\n[ERREUR] Dossier source introuvable : {source_dir}\n")
        return

    traites      = set()   # noms normalisés déjà traités
    stats        = {}      # compteurs par sous-dossier
    introuvables = []      # fichiers listés mais absents du dossier

    print("\n" + "═" * 68)
    print(f"  Organisation KB ML/DS  —  {'DRY-RUN (simulation)' if dry_run else 'MODE RÉEL'}")
    print(f"  Source : {source_dir}")
    print("═" * 68)

    for dossier_principal, sous_dossiers in STRUCTURE.items():
        icone = ICONES.get(dossier_principal, "📁")
        total_attendu = sum(len(v) for v in sous_dossiers.values())
        print(f"\n{'─'*68}")
        print(f"  {icone}  {dossier_principal}/  ({total_attendu} fichiers attendus)")
        print(f"{'─'*68}")

        for sous_dossier, fichiers in sous_dossiers.items():
            dst = source_dir / dossier_principal / sous_dossier
            if not dry_run:
                dst.mkdir(parents=True, exist_ok=True)

            label = f"{dossier_principal}/{sous_dossier}"
            stats[label] = {"ok": 0, "introuvable": 0, "deja_traite": 0}

            print(f"\n    📂 {sous_dossier}/")

            for nom in fichiers:
                f = find_file(source_dir, nom)
                if f is None:
                    print(f"      [INTROUVABLE]  {nom}")
                    stats[label]["introuvable"] += 1
                    introuvables.append((label, nom))
                    continue
                if normalise(f.name) in traites:
                    print(f"      [DÉJÀ TRAITÉ]  {f.name}")
                    stats[label]["deja_traite"] += 1
                    continue
                if move_file(f, dst, dry_run):
                    traites.add(normalise(f.name))
                    stats[label]["ok"] += 1

    # ── Rapport final ──────────────────────────────────────────
    total_ok = sum(v["ok"]          for v in stats.values())
    total_nf = sum(v["introuvable"] for v in stats.values())
    total_dt = sum(v["deja_traite"] for v in stats.values())

    print("\n\n" + "═" * 68)
    print("  RAPPORT FINAL")
    print("═" * 68)

    current_parent = None
    for label, s in stats.items():
        parent, sous = label.split("/", 1)
        if parent != current_parent:
            icone = ICONES.get(parent, "📁")
            print(f"\n  {icone}  {parent}/")
            current_parent = parent
        nf = f"  ⚠ {s['introuvable']} introuvables" if s["introuvable"] else ""
        dt = f"  ({s['deja_traite']} déjà traités)" if s["deja_traite"] else ""
        print(f"       📂 {sous:<45}  {s['ok']:>3} ok{nf}{dt}")

    print("\n" + "─" * 68)
    print(f"  Total déplacés     : {total_ok} fichiers")
    if total_nf:
        print(f"  Total introuvables : {total_nf} fichiers")
        print("\n  Détail introuvables (vérifier les noms sur le Drive) :")
        for (label, nom) in introuvables:
            print(f"    • [{label}]  {nom}")
    if total_dt:
        print(f"  Déjà traités (ignorés) : {total_dt}")
    print("═" * 68)
    if dry_run:
        print("  ⚠️   Mode DRY-RUN actif : aucun fichier n'a été déplacé.\n")
    else:
        print("  ✅  Organisation terminée avec succès.\n")

# ═══════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Organise les PDFs ML/DS en dossiers et sous-dossiers thématiques."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Chemin vers le dossier contenant tous les PDFs"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simule les déplacements sans toucher aux fichiers"
    )
    args = parser.parse_args()
    organise(Path(args.source), dry_run=args.dry_run)

if __name__ == "__main__":
    main()
