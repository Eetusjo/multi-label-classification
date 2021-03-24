
# FinCORE labels heirarchy
FC_CAT_UPPER = {
    "NA": "narrative",
    "OP": "opinion",
    "IN": "informational descript.",
    "ID": "discussion",
    "HI": "how-to",
    "IG": "persuation",
    "LY": "lyrical",
    "SP": "spoken",
    "OS": "other"
}
FC_CAT_UPPER_L2I = {
    "NA": 0,
    "OP": 1,
    "IN": 2,
    "ID": 3,
    "HI": 4,
    "IG": 5,
    "LY": 6,
    "SP": 7,
    "OS": 8
}
# FC_CAT_LOWER = {
#     "NE": "news"
#     "SR": "sports"
#     "PB": "personal blog"
#     HA N    A Historical article
#     FC NA Fiction
#     TB NA Travel blog
#     CB NA Community blogs
#     OA NA Online article
#     OP Opinion
#
#     OB OP Personal opinion blogs
#     RV OP Reviews
#     RS OP Religious blogs/sermons
#     AV OP Advice
#     IN Informational description
#
#     JD IN Job description
#     FA IN FAQs
#     DT IN Description of a thing
#     IB IN Information blogs
#     DP IN Description of a person
#     RA IN Research articles
#     LT IN Legal terms / conditions
#     CM IN Course materials
#     EN IN Encyclopedia articles
#     RP IN Report
#     ID Interactive discussion
#
#     DF ID Discussion forums
#     QA ID Question-answer forums
#     HI How-to/instructions
#
#     RE HI Recipes
#     IP IG Informational persuasion
#
#     DS IG Description with intent to sell
#     EB IG News-opinion blogs / editorials
#     Lyrical LY
#
#     PO LY Poems
#     SL LY Songs
#     Spoken SP
#
#     IT SP Interviews
#     FS SP Formal speeches
#     Others OS
#
#     MT OS Machine-translated / generated texts
# }

def fincore_tags_to_onehot(tags):
    "Convert list of FinCORE tags to a one-hot encoded vector."
    l = [0]*len(FC_CAT_UPPER_L2I.keys())
    for tag in tags:
        l[FC_CAT_UPPER_L2I[tag]] = 1
    return l


def fincore_to_dict_upper(path, id_prefix):
    """Convert FinCORE samples to dict format with only high-level tags."""
    data = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            try:
                tags, text = line.strip().split("\t")
            except:
                continue

            tags = [t for t in tags.split() if t in FC_CAT_UPPER.keys()]

            data.append({
                "id": f"{id_prefix}-{i}",
                "tags": tags,
                "num_tags": len(tags),
                "labels": fincore_tags_to_onehot(tags),
                "text": text.strip()
            })

    return data
