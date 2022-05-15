from fact_summac.model_summac import SummaCZS

def classify(document, summary):
    model = SummaCZS(granularity="sentence", model_name="vitc")

    # INPUT EXAMPLE
    document = """Scientists are studying Mars to learn about the Red Planet and find landing sites for future missions. One possible site, known as Arcadia Planitia, is covered instrange sinuous features. The shapes could be signs that the area is actually made of glaciers, which are large masses of slow-moving ice. Arcadia Planitia is in Mars' northern lowlands."""
    summary = "There are strange shape patterns on Arcadia Planitia. The shapes could indicate the area might be made of glaciers. This makes Arcadia Planitia ideal for future missions."

    score = model.score([document], [summary])
    if (score <= 0):
        return 0
    else:
        return score
