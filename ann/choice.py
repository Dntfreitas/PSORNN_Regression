from ann.annRedWine import ANNRedWine


def getAnn(id):
    if id == 1:
        return ANNRedWine()
    else:
        return None
