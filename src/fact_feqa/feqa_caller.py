import sys

HOME_REPO = "/home/lr/faza.thirafi/raid/repository-kenkyuu-models/"
HOME_DATA = "/home/lr/faza.thirafi/raid_elmo/cache/"

sys.path.insert(0, HOME_REPO + "feqa/")
# import benepar, nltk
# benepar.download('benepar_en2')
# nltk.download('stopwords')
from feqa import FEQA


def classify(document, summary):
    scorer = FEQA(use_gpu=True)
    # INPUT EXAMPLE
    # document = """Scientists are studying Mars to learn about the Red Planet and find landing sites for future missions. One possible site, known as Arcadia Planitia, is covered instrange sinuous features. The shapes could be signs that the area is actually made of glaciers, which are large masses of slow-moving ice. Arcadia Planitia is in Mars' northern lowlands."""
    # summary = "There are strange shape patterns on Arcadia Planitia. The shapes could indicate the area might be made of glaciers. This makes Arcadia Planitia ideal for future missions."

    scores = scorer.compute_score([document], [summary], aggregate=False)
    return scores[0]